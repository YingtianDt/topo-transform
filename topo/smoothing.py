import torch


# borrow from: https://github.com/epflneuroailab/topolm/blob/main/eval/hrf.py
class NeuronSmoothing:
    def __init__(self, fwhm_mm=2, resolution_mm=1, kernel_radius_multiplier=3.0,
                 chunk_size=1024, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.fwhm_mm = fwhm_mm
        self.resolution_mm = resolution_mm
        self.kernel_radius_multiplier = kernel_radius_multiplier
        self.chunk_size = chunk_size
        self.device = device

    def __call__(self, activations, positions):
        """
        Args:
            positions: torch.Tensor of shape (n_neurons, 2) with (x, y) coordinates
            activations: torch.Tensor of shape (n_samples, n_neurons) with activation values
        
        Returns:
            features_smoothed: torch.Tensor of shape (n_samples, n_grid_points) with smoothed activations
            grid_positions: torch.Tensor of shape (n_grid_points, 2) with grid coordinates
        """
        has_time = len(activations.shape) == 5
        B = activations.shape[0]
        if has_time:
            T = activations.shape[1]
            activations = activations.reshape(B * T, -1)  # (B*T, n_neurons)
        else:
            activations = activations.reshape(B, -1)  # (B, n_neurons)

        # Ensure tensors are on the correct device
        if not isinstance(positions, torch.Tensor):
            positions = torch.tensor(positions, dtype=torch.float32, device=self.device)
        else:
            positions = positions.to(self.device)
            
        if not isinstance(activations, torch.Tensor):
            activations = torch.tensor(activations, dtype=torch.float32, device=self.device)
        else:
            activations = activations.to(self.device)

        # Get grid coordinates
        xmin, xmax = torch.floor(positions[:, 0].min()).item(), torch.ceil(positions[:, 0].max()).item()
        ymin, ymax = torch.floor(positions[:, 1].min()).item(), torch.ceil(positions[:, 1].max()).item()
        
        grid_x = torch.arange(xmin, xmax + 1, self.resolution_mm, device=self.device)
        grid_y = torch.arange(ymin, ymax + 1, self.resolution_mm, device=self.device)

        grid_h = len(grid_y)
        grid_w = len(grid_x)
        
        grids = torch.meshgrid(grid_x, grid_y, indexing='ij')
        grid_positions = torch.stack([grids[0].flatten(), grids[1].flatten()], dim=1)  # (n_grid_points, 2)

        # Compute sigma and kernel radius
        sigma = self.fwhm_mm / (2.0 * torch.sqrt(2.0 * torch.log(torch.tensor(2.0))))
        kernel_radius = self.kernel_radius_multiplier * sigma
        kernel_radius_sq = kernel_radius ** 2

        # Pre-compute Gaussian normalization constant
        gauss_norm = 1. / (2 * torch.pi * sigma ** 2)
        sigma_sq_inv = 1. / (2 * sigma ** 2)

        n_grid_points = grid_positions.shape[0]
        if has_time:
            features_smoothed = torch.zeros((B * T, n_grid_points), device=self.device)
        else:
            features_smoothed = torch.zeros((B, n_grid_points), device=self.device)

        # Process grid points in chunks
        for chunk_start in range(0, n_grid_points, self.chunk_size):
            chunk_end = min(chunk_start + self.chunk_size, n_grid_points)
            grid_chunk = grid_positions[chunk_start:chunk_end]  # (chunk_size, 2)
            chunk_len = chunk_end - chunk_start
            
            # Compute squared distances: (chunk_size, n_neurons)
            # Broadcasting: (chunk_size, 1) - (n_neurons,) -> (chunk_size, n_neurons)
            dx = grid_chunk[:, 0:1] - positions[:, 0].unsqueeze(0)  # (chunk_size, n_neurons)
            dy = grid_chunk[:, 1:2] - positions[:, 1].unsqueeze(0)  # (chunk_size, n_neurons)
            d_square = dx ** 2 + dy ** 2
            
            # Create mask for neurons within kernel radius
            mask = d_square <= kernel_radius_sq  # (chunk_size, n_neurons)
            
            # Compute Gaussian weights (set to 0 outside radius)
            weights = torch.zeros_like(d_square)
            weights[mask] = gauss_norm * torch.exp(-d_square[mask] * sigma_sq_inv)
            
            # Apply smoothing: (B, n_neurons) @ (n_neurons, chunk_size) -> (B, chunk_size)
            # weights is (chunk_size, n_neurons), so weights.T is (n_neurons, chunk_size)
            features_smoothed[:, chunk_start:chunk_end] = activations @ weights.T

        grid_dims = (1, grid_h, grid_w)
        if has_time:
            features_smoothed = features_smoothed.reshape(B, T, *grid_dims)  # (B, T, 1, H, W)
        else:
            features_smoothed = features_smoothed.reshape(B, *grid_dims)  # (B, 1, H, W)


        return features_smoothed, grid_positions, grid_dims