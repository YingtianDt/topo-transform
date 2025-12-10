from .mvit.mvitv1 import MViTV1
from .uniformer import UniFormer
from .vjepa import VJEPA, VJEPASwapopt
from .tdann import TDANN

from torchvision import transforms

vit_transform = transforms.Compose([
    transforms.Resize((224, 224)), 
    transforms.Lambda(lambda img: img/255.0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])    