import streamlit as st
from transformers import CLIPProcessor, CLIPModel
from PIL import Image

from matplotlib import pyplot as plt
import torch
import numpy as np

from argparse import ArgumentParser
import heapq


def parse_args() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument('--flower_species_list', type=str, default='flowers.txt')
    parser.add_argument('--img_dir', type=str, default='flower_imgs')
    return parser.parse_args()

def read_flower_species_list(path: str) -> list[str]:
    with open(path, mode='r') as f:
        return [line.strip().lower() for line in f]

def dist(e1: torch.Tensor, e2: torch.Tensor) -> torch.Tensor:
    return torch.relu(1.0 - torch.matmul(e1, e2.T))

@st.cache_resource
def load_model() -> tuple[CLIPModel, CLIPProcessor]:
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor

def dijkstra(start: int, end: int, adj_mat: list[list[float]]) -> tuple[float, list[int]]:
    """
    Find the shortest path between two nodes in a graph using Dijkstra's algorithm.
    """
    n = len(adj_mat)
    cost = [float('inf')] * n
    parent = [None] * n
    cost[start] = 0
    visited = [False] * n
    queue = [(0, start)]
    while queue:
        _, u = heapq.heappop(queue)
        if visited[u]: continue
        visited[u] = True
        for v in range(n):
            if visited[v]: continue
            if cost[v] > cost[u] + adj_mat[u][v]:
                cost[v] = cost[u] + adj_mat[u][v]
                parent[v] = u
                heapq.heappush(queue, (cost[v], v))
    path = []
    u = end
    while u is not None:
        path.append(u)
        u = parent[u]
    path.reverse()
    return cost[end], path

def find_hops(start: str, end: str, flowers: dict[str, Image.Image]) -> list[str]:
    model, processor = load_model()
    with torch.no_grad():
        output = model(**processor(
            text=['UNUSED'],
            images=list(flowers.values()),
            return_tensors="pt",
            padding=True
        ))
    _, path = dijkstra(
        start=list(flowers.keys()).index(start),
        end=list(flowers.keys()).index(end),
        adj_mat=dist(output.image_embeds, output.image_embeds).tolist()
    )
    return np.take(list(flowers.keys()), path)

if __name__ == '__main__':
    args = parse_args()

    st.title('Flower Hopper')
    flower_species = read_flower_species_list(args.flower_species_list)
    flower_A = st.selectbox(
        'Flower A:',
        flower_species,
        index=None,
        placeholder="Select one flower...",
    )
    flower_B = st.selectbox(
        'Flower B:',
        flower_species,
        index=None,
        placeholder="Select one flower...",
    )
    if flower_A and flower_B and st.button(label='Find Hops'):
        path = find_hops(
            start=flower_A,
            end=flower_B,
            flowers={s: Image.open(f'{args.img_dir}/{s}.jpg') for s in read_flower_species_list(args.flower_species_list)}
        )
        ncols = 3
        nrows = len(path) // ncols + int(len(path) % ncols > 0)
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols)
        fig.suptitle('Flower Hops')
        for i, ax in enumerate(axes.flat):
            if i < len(path):
                ax.imshow(np.asarray(Image.open(f'{args.img_dir}/{path[i]}.jpg')))
                if i == 0:
                    ax.set_title(f'Start: {path[i]}')
                elif i == len(path) - 1:
                    ax.set_title(f'End: {path[i]}')
                else:
                    ax.set_title(f'Hop {i}: {path[i]}')
            ax.axis('off')
        st.pyplot(fig)
