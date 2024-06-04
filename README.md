## What is this?

This is an attempt to prototype a minimal analog to the [Candle](https://github.com/huggingface/candle) ML framework from Huggingface with an extra layer of type safety provided by Rust const generics. Like in [PyTorch](https://pytorch.org/) or [pandas](https://pandas.pydata.org/), everything in Candle has type `Tensor`, with dimension analysis handled via runtime assertions. Let's see how much of this can be pushed to compile time and what the resulting DX can look like.

> [!WARNING]
> The code in this repository is very WIP and has mostly pedagogical value at the moment.
