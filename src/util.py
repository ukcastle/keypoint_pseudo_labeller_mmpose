import time

def timeit(func):
  def wrapper(*args, **kwargs):
    s = time.time()
    result = func(*args, **kwargs)
    e = time.time()

    print(f"function : {func.__name__} | time : {(e-s):2.4f}")

    return result
  return wrapper