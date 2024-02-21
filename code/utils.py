import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import PIL.Image as Image
import numpy as np
import torch


def show_tensor_image(tensor_image: torch.Tensor):
    pil_image = transforms.ToPILImage()(tensor_image)
    plt.imshow(pil_image, cmap='gray')
    plt.show()


def show_numpy_image(np_image: np.ndarray):
    im = Image.fromarray(np_image.astype(np.uint8))
    plt.imshow(im, cmap='gray')
    plt.show()


def image_to_tensor(image_path: str, device: str) -> torch.Tensor:
    image = Image.open(image_path).convert('RGB')

    tensor_image = transforms.ToTensor()(image)
    tensor_image = tensor_image.to(device)

    return tensor_image


def add_fake_noise(image_tensor, noise_goofy=0, device: str = "cpu"):
    dropout_prob_mask = torch.sum(image_tensor, dim=0) / 3
    random_decision_mask = torch.rand(dropout_prob_mask.shape, device=device)
    dropout_mask = dropout_prob_mask - noise_goofy > random_decision_mask
    
    return dropout_mask == False, image_tensor * dropout_mask


def get_images_and_masks(path: str, device: str):
    dev_img = image_to_tensor(path, device=device)
    marked_for_denoising_dev, noisy_image_dev = add_fake_noise(dev_img, noise_goofy=0.0, device=device)
    marked_for_denoising_dev = marked_for_denoising_dev.unsqueeze(0).to(torch.float32)
    unmarked_for_denoising_dev = (marked_for_denoising_dev == 0).to(torch.float32)
    return (dev_img, noisy_image_dev, marked_for_denoising_dev, unmarked_for_denoising_dev)