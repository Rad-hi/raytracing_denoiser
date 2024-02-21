import torch
import torch.nn.functional as F
import math
import time
import numpy as np
import numba
from IPython.display import clear_output

from utils import *


# Falgs to toggle what to benchmark (TODO[RS]: do proper benchmarking)
VANILLA = True
NUMBA = True
SLOW = False
CONV = True


# TODO[RS]: make into some proper constants
size = 3
sigma = 1
kernel_weights = torch.tensor([[math.exp(-((i - size//2)**2 + (j - size//2)**2) / (2 * sigma**2)) for j in range(size)] for i in range(size)], dtype=torch.float32, device='cuda:0')
kernel = kernel_weights.repeat(1, 1, 1, 1)
padding = (kernel.shape[2] - 1) // 2


def loop_bloom(image_tensor, marked_for_denoising, unmarked_for_denoising):
    image_tensor = image_tensor.permute(1, 2, 0)
    result = torch.zeros(image_tensor.shape, dtype=torch.float32, device='cpu')
    
    for y in range(image_tensor.shape[0]):
        for x in range(image_tensor.shape[1]):

            target_pixel = image_tensor[y][x]
            
            if unmarked_for_denoising[0][y][x]:
                result[y][x] = target_pixel
                
                if y + 1 < image_tensor.shape[0] and marked_for_denoising[0][y + 1][x]:
                    result[y + 1][x] = target_pixel
                
                if y - 1 >= 0 and marked_for_denoising[0][y - 1][x]:
                    result[y - 1][x] = target_pixel
                
                if x - 1 >= 0 and marked_for_denoising[0][y][x - 1]:
                    result[y][x - 1] = target_pixel
                
                if x + 1 < image_tensor.shape[1] and marked_for_denoising[0][y][x + 1]:
                    result[y][x + 1] = target_pixel

    result = result.permute(2, 0, 1)
    
    return result


def loop_bloom_vanilla(image, marked_for_denoising, unmarked_for_denoising):

    result = np.zeros(image.shape, dtype=np.float32)
    h, w = image.shape[:2]

    for y in range(1, h - 1):
        for x in range(1, w - 1):

            target_pixel = image[y][x]
            
            if unmarked_for_denoising[y][x]:
                result[y][x] = target_pixel

                if marked_for_denoising[y + 1][x]:
                    result[y + 1][x] = target_pixel

                if marked_for_denoising[y - 1][x]:
                    result[y - 1][x] = target_pixel
                
                if marked_for_denoising[y][x - 1]:
                    result[y][x - 1] = target_pixel

                if marked_for_denoising[y][x + 1]:
                    result[y][x + 1] = target_pixel

    return result


loop_bloom_numba = numba.njit(cache=True)(loop_bloom_vanilla)


# this function is so smart your python interpreter might have trouble understanding it
def bloom(image_tensor, marked_for_denoising, unmarked_for_denoising, kernel=kernel, padding=padding):
    # explanation why goofy ahh channel split and not just conv over full image:
    # for some reason when i do it the normal way the image comes out black and white and this just works im not gonna fix it
    red_channel = image_tensor[0].unsqueeze(0)
    green_channel = image_tensor[1].unsqueeze(0)
    blue_channel = image_tensor[2].unsqueeze(0)
    
    # apply padding seperatly because i want to use "reflect" mode
    red_channel_padded = F.pad(red_channel, [padding] * 4, mode='reflect')
    green_channel_padded = F.pad(green_channel, [padding] * 4, mode='reflect')
    blue_channel_padded = F.pad(blue_channel, [padding] * 4, mode='reflect')

    # apply conv here
    red_conv = F.conv2d(red_channel_padded, kernel)
    green_conv = F.conv2d(green_channel_padded, kernel)
    blue_conv = F.conv2d(blue_channel_padded, kernel)

    # stack the channels back into a single image
    colored_result = torch.cat((red_conv, green_conv, blue_conv), dim=0)

    # this image is the new information that the function has generated
    black_fill = colored_result * marked_for_denoising
    
    # we want to normalize that because conv might return a bigger value than 1
    black_fill /= torch.max(black_fill)
    
    # original image is the image as if this function has never touched it
    # even reset previous of this function's edurations over it
    original_image = image_tensor * unmarked_for_denoising
    
    # place the progress of this function + all the functions before it on the original image
    return original_image + black_fill

def denoise(image_tensor, blooming_func, marked_for_denoising, unmarked_for_denoising, difference_tolerance_precent, movie=False, movie_fps=10):
    difference = difference_tolerance_precent + 1

    while difference > difference_tolerance_precent:
        movie_time_start = time.time()
        before_denoise = image_tensor
        image_tensor = blooming_func(image_tensor, marked_for_denoising, unmarked_for_denoising)
        difference = torch.sum(torch.abs(before_denoise - image_tensor)) / torch.prod(torch.tensor(image_tensor.shape)) * 100
        
        if movie:
            clear_output(wait=True)
            show_tensor_image(image_tensor)
            movie_time_end = time.time()
            
            # make sure that we wait the right amount of time before the next frame is drawn
            time.sleep(max(0, 1 / movie_fps - (movie_time_end - movie_time_start)))
    
    return image_tensor


def main():

    PATH = "/home/rs/Desktop/raytracing_denoiser/example_images/horse.png"
    (cpu_image, noisy_image_cpu, marked_for_denoising_cpu, unmarked_for_denoising_cpu) = get_images_and_masks(PATH, "cpu")
    (gpu_image, noisy_image_gpu, marked_for_denoising_gpu, unmarked_for_denoising_gpu) = get_images_and_masks(PATH, "cuda")


    if CONV:
        ts = time.perf_counter()
        image_0 = denoise(noisy_image_gpu, bloom, marked_for_denoising_gpu, unmarked_for_denoising_gpu, difference_tolerance_precent=0.05, movie=False, movie_fps=60)
        took = time.perf_counter() - ts
        print(f'Denoiser [bloom] took: {took * 1000.:.2f} ms')

    if SLOW:
        ts = time.perf_counter()
        image_1 = denoise(noisy_image_cpu, loop_bloom, marked_for_denoising_cpu, unmarked_for_denoising_cpu, difference_tolerance_precent=0.05, movie=False, movie_fps=60)
        took = time.perf_counter() - ts
        print(f'Denoiser [loop_bloom] took: {took:.2f} s')


    # Read everythin as numpy arrays
    pil_image = transforms.ToPILImage()(cpu_image)
    in_0 = np.array(pil_image)
    in_1 = marked_for_denoising_cpu.permute(1, 2, 0).numpy()
    in_2 = unmarked_for_denoising_cpu.permute(1, 2, 0).numpy()

    # TODO[RS]: make work with denoise --> improve denoising performance
    if NUMBA:
        ts = time.perf_counter()
        image_2 = loop_bloom_numba(in_0, in_1, in_2)
        took = time.perf_counter() - ts
        print(f'Denoiser [loop_bloom_numba] took: {took * 1000.:.2f} ms')

    if VANILLA:
        ts = time.perf_counter()
        image_2 = loop_bloom_vanilla(in_0, in_1, in_2)
        took = time.perf_counter() - ts
        print(f'Denoiser [loop_bloom_vanilla] took: {took * 1000.:.2f} ms')


    # TODO[RS]: make into a single graph (subplots)
    print("original image:")
    show_tensor_image(gpu_image)
    print("noisy image:")
    show_tensor_image(noisy_image_gpu)
    print("black pixels -> denoiser needs to guess:")
    show_tensor_image(unmarked_for_denoising_gpu)

    if CONV:
        print("bloom output")
        show_tensor_image(image_0)
    if SLOW:
        print("slow-loop-bloom output")
        show_tensor_image(image_1)
    if VANILLA or NUMBA:
        print("fast-loop-bloom output")
        show_numpy_image(image_2)


if __name__ == '__main__':
    main() 