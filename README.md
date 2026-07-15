# Cuda_code

Minimal CUDA example that processes each test case independently.

## Build

```bash
nvcc case_processor.cu -O2 -o case_processor
```

## Run

Input format:

- First value: number of test cases `T`
- For each case: `N` followed by `N` integers

The program squares each integer on the GPU and prints one output line per case.

Example:

```text
Input
2
3 1 2 3
4 4 5 6 7

Output
Case #1: 1 4 9
Case #2: 16 25 36 49
```
