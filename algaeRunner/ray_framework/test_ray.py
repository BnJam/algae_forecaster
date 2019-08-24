import ray
import time

@ray.remote
def square(n):
    return n*n


if __name__ == "__main__":
    ray.init()
    future = square.remote(5)
    print(ray.get(future))