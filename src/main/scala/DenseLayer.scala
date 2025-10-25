
import chisel3._
import chisel3.util.Cat
import Activations._

/* 
  A fully connected dense layer with x inputs and n outputs (neurons)
  Each neuron performs a MAC of the inputs and weights, adds a bias, 
  applies an activation function, and saturates to the input width.
  
  Parameters:
    x: number of inputs
    in_w: bit-width of inputs, weights, biases, and outputs
    n: number of neurons (outputs)
    af: activation function (0=none, 1=ReLU, 2=tanh, 3=sigmoid)
    fracBits: number of fractional bits in fixed-point representation

  I/O:
    in:  vector of x inputs (SInt)
    w:   matrix of weights (n x x) (SInt)
    b:   vector of biases (n) (SInt)
    out: vector of n outputs (SInt)
*/ 

class DenseLayer(x: Int = 2, in_w: Int = 8, n: Int = 2, af: Int = 1, fracBits: Int = 6) extends Module {
  val io = IO(new Bundle {
    val in  = Input(Vec(x, SInt(in_w.W)))
    val w = Input(Vec(n, Vec(x, SInt(in_w.W))))
    val b = Input(Vec(n, SInt(in_w.W))) 
    val out = Output(Vec(n, SInt(in_w.W)))
  })

  // Saturate a value to a target signed width (in_w)
  private def satToWidth(v: SInt, outW: Int): SInt = {
    val max = ((1 << (outW - 1)) - 1).S(v.getWidth.W)
    val min = (-(1 << (outW - 1))).S(v.getWidth.W)
    val clamped = Mux(v > max, max, Mux(v < min, min, v))
    clamped.asTypeOf(SInt(outW.W))
  }

  // Compute per-neuron MAC, apply activation, then saturate to output width
  for (j <- 0 until n) {
    val products = (0 until x).map { i =>
      // Multiply in fixed-point and downscale back by fracBits
      (io.in(i) * io.w(j)(i)) >> fracBits
    }
    val sum = products.reduce(_ +& _) +& io.b(j)

    val activated: SInt = af match {
      case 0 => sum
      case 1 => relu(sum)
      case 2 => tanh(sum, fracBits)
      case 3 => sigmoid(sum, fracBits)
      case _ => sum
    }

    io.out(j) := satToWidth(activated, in_w)
  }

}
