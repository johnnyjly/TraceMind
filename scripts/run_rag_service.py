"""Run a prewarmed local retrieval service separately from API and worker."""
import os
import logging

# Bound local CPU inference before importing the numerical libraries.
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')


def main():
    import torch
    import uvicorn
    from tracemind.rag.query_service import create_app

    logging.basicConfig(level=logging.INFO)
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    uvicorn.run(create_app(), host='127.0.0.1', port=8011)


if __name__ == '__main__':
    main()
