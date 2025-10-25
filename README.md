# Neural Network: Dense Layer (Chisel)

- Minimal fixed-point dense (fully connected) layer in Chisel.
- Building block for feed-forward neural networks (MLPs).
- Supports parameterized size, fixed-point scaling, and common activations.
- Includes quantization helpers and unit tests with chiseltest.

## Module

- Class: `DenseLayer(x: Int = 2, in_w: Int = 8, n: Int = 2, af: Int = 1, fracBits: Int = 6)`
- File: `src/main/scala/DenseLayer.scala`
- I/O:
  - `io.in: Vec(x, SInt(in_w.W))` - input vector
  - `io.w: Vec(n, Vec(x, SInt(in_w.W)))` - weights (n by x)
  - `io.b: Vec(n, SInt(in_w.W))` - biases
  - `io.out: Vec(n, SInt(in_w.W))` - output vector
- Parameters:
  - `x` - input dimension
  - `in_w` - bit width for inputs/weights/biases/outputs (signed)
  - `n` - number of neurons (outputs)
  - `af` - activation: 0=none, 1=ReLU, 2=tanh (hard), 3=sigmoid (hard)
  - `fracBits` - fractional bits for fixed-point interpretation

## Behavior

- Per-neuron compute: `sum = (sum_i (in[i] * w[j][i]) >> fracBits) + b[j]`
- Activation: applied per `af` parameter via `Activations` helpers.
- Saturation: results are clamped to `in_w` signed range on output.
- Fixed-point: values are SInts interpreted as real * 2^fracBits (Q format).

## Fixed-Point & Activations

- Fixed-point convention:
  - Scale factor: `1 << fracBits`
  - Example (fracBits=6): 1.0 <-> 64, -1.0 <-> -64
- Activations (hardware-friendly approximations):
  - ReLU: `max(0, x)`
  - HardTanh: clamps to [-1.0, +1.0] in fixed-point
  - HardSigmoid: `clamp(0.25*x + 0.5, 0, 1)` in fixed-point
- See: `src/main/scala/Activations.scala`

## Quantization Helpers

- Convert real values to fixed-point SInt at elaboration:
  - `Quantization.toFixedSInt(value: Double|Int, in_w, fracBits): SInt`
  - `Quantization.toFixedVec(seq[Double], in_w, fracBits): Vec[SInt]`
  - `Quantization.toFixedMatrix(seq[seq[Double]], in_w, fracBits): Vec[Vec[SInt]]`
- Debug aid:
  - `Quantization.fromFixed(value: SInt, fracBits): Double` (elaboration-time only)
- File: `src/main/scala/Quantization.scala`

## Project Layout

- Sources:
  - `src/main/scala/DenseLayer.scala`
  - `src/main/scala/Activations.scala`
  - `src/main/scala/Quantization.scala`
- Tests:
  - `src/test/scala/DenseLayerSpec.scala`
- Build:
  - `build.sbt` (Scala 2.13.16, Chisel 6.7.0, chiseltest 6.0.0)
  - `project/build.properties` (sbt 1.11.6)
- CI:
  - `.github/workflows/ci.yml` (runs `sbt test` with JDK 17)

## Build & Test

- Requirements:
  - JDK 17+ (CI uses 17.0.14)
  - sbt 1.11.x
- Run:
  - `cd ChiselLearning/neural-network`
  - `sbt test`
- Tests cover:
  - Saturation with identity activation
  - HardTanh clamp within [-1, +1]
  - ReLU zeroing negatives

## Compose Multiple Layers (MLP)

Stack multiple `DenseLayer`s to form a feed-forward network. Example: 3->4->2 MLP with ReLU in the hidden layer and identity at the output.

```scala
import chisel3._
import Quantization._

class MLP extends Module {
  val io = IO(new Bundle {
    val in  = Input(Vec(3, SInt(8.W)))
    val out = Output(Vec(2, SInt(8.W)))
  })

  val fracBits = 6
  val l1 = Module(new DenseLayer(x = 3, in_w = 8, n = 4, af = 1, fracBits = fracBits)) // ReLU
  val l2 = Module(new DenseLayer(x = 4, in_w = 8, n = 2, af = 0, fracBits = fracBits)) // identity

  l1.io.in := io.in
  l2.io.in := l1.io.out

  // Provide weights/biases at elaboration time (example literals shown)
  def setWb(layer: DenseLayer, w: Seq[Seq[Double]], b: Seq[Double]): Unit = {
    for (j <- w.indices; i <- w(j).indices) layer.io.w(j)(i) := toFixedSInt(w(j)(i), 8, fracBits)
    for (j <- b.indices) layer.io.b(j) := toFixedSInt(b(j), 8, fracBits)
  }

  setWb(l1, Seq.fill(4)(Seq(0.2, -0.1, 0.3)), Seq(0.0, 0.0, 0.0, 0.0))
  setWb(l2, Seq(Seq(0.5, -0.2, 0.1, 0.4), Seq(-0.3, 0.6, 0.2, -0.1)), Seq(0.0, 0.0))

  io.out := l2.io.out
}
```

## Quick Example

Instantiate a 2-input, 2-output dense layer with ReLU and Q1.6 scaling:

```scala
import chisel3._
import Quantization._

class Top extends Module {
  val io = IO(new Bundle {
    val in  = Input(Vec(2, SInt(8.W)))
    val out = Output(Vec(2, SInt(8.W)))
  })

  val layer = Module(new DenseLayer(x = 2, in_w = 8, n = 2, af = 1, fracBits = 6))
  layer.io.in := io.in

  // Example params: weights and biases elaboration-time literals
  val w = Seq(Seq(1.0, -0.5), Seq(0.75, 0.25)).map(_.map(toFixedSInt(_, 8, 6)))
  val b = Seq(0.0, 0.0).map(toFixedSInt(_, 8, 6))

  for (j <- 0 until 2; i <- 0 until 2) {
    layer.io.w(j)(i) := w(j)(i)
  }
  for (j <- 0 until 2) {
    layer.io.b(j) := b(j)
  }

  io.out := layer.io.out
}
```

## Notes & Limits

- `in_w` sets the width for all IOs; outputs saturate to this width.
- Use appropriate `fracBits` for your dynamic range; large `fracBits` increases risk of overflow before saturation.
- Feed-forward only: no recurrence/state across cycles; weights/biases are provided externally (e.g., literals or ports).
- Single-cycle combinational MAC per layer; add pipeline registers if timing requires.
- When composing layers, keep `in_w`/`fracBits` consistent across layers or insert conversion as needed.

## License
Created by RifkiFi

