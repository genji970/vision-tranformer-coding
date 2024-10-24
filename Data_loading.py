import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

def data_loader():
    # 1. 데이터 전처리 정의
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # ViT는 보통 224x224 크기의 입력을 사용합니다.
    transforms.ToTensor(),          # 이미지를 Tensor로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 정규화
    ])

    # 2. CIFAR-10 데이터셋 다운로드 및 로딩
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )

    # 3. 서브셋 인덱스 정의 (예: 처음 1000개만 사용)
    subset_indices = list(range(1000))  # 처음 1000개 데이터만 사용
    train_subset = Subset(train_dataset, subset_indices)

    # 3. DataLoader 생성 (배치 크기 32)
    train_loader = DataLoader(train_subset, batch_size=32, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    return train_loader , test_loader