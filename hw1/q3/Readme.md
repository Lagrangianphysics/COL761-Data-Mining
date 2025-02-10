# README

## Remark

This project uses a C++ executable `Multi_thread_extract_unique_graphs.out` for graph isomorphism in the `identify_features.py` script. We have implemented this code ourselves.

Ensure that `Multi_thread_extract_unique_graphs.out` is in the current directory.

## Troubleshooting

If you encounter any issues running the executable (which we don't anticipate), follow these steps:

1. Navigate to the `cpp_files` directory.
2. Compile `Multi_thread_extract_unique_graphs.cpp` using the following command:
    ```sh
    g++ Multi_thread_extract_unique_graphs.cpp
    ```
3. An output file `a.out` will be generated.
4. Rename `a.out` to `Multi_thread_extract_unique_graphs.out`:
    ```sh
    mv a.out Multi_thread_extract_unique_graphs.out
    ```
5. Copy `Multi_thread_extract_unique_graphs.out` to the current directory.
6. Run the `identify_features.py` script again.

