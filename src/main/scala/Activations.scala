import chisel3._
import chisel3.util._

/*
  Hardware-friendly activation function helpers.
  - relu / reluS
  - tanh (hard tanh approximation)
  - sigmoid (hard sigmoid approximation)

  Fixed-point note:
  - Treat signals as fixed-point with 'fracBits' fractional bits.
  - 1.0 is represented as (1 << fracBits).
*/

object Activations {
  // ReLU: max(0, x)
  def reluS(v: SInt): SInt = Mux(v >= 0.S, v, 0.S(v.getWidth.W))
  // SInt-friendly overload
  def relu(v: SInt): SInt = reluS(v)
  // UInt wrapper
  def relu(v: UInt): UInt = reluS(v.asSInt).asUInt

  // HardTanh: clamp to [-1.0, 1.0] in fixed-point
  def tanhHardS(v: SInt, fracBits: Int): SInt = {
    val max = (1 << fracBits).S(v.getWidth.W)      // +1.0
    val min = (-(1 << fracBits)).S(v.getWidth.W)   // -1.0
    Mux(v > max, max, Mux(v < min, min, v))
  }
  // SInt-friendly overload
  def tanh(v: SInt, fracBits: Int): SInt = tanhHardS(v, fracBits)
  // UInt wrapper
  def tanh(v: UInt, fracBits: Int): UInt = tanhHardS(v.asSInt, fracBits).asUInt

  // HardSigmoid: y = clamp(0.25*x + 0.5, 0, 1) in fixed-point
  def sigmoidHardS(v: SInt, fracBits: Int): SInt = {
    val scale = 1 << fracBits                      // 1.0
    val half  = (scale >> 1).S(v.getWidth.W)       // 0.5
    val one   = scale.S(v.getWidth.W)              // 1.0
    val zero  = 0.S(v.getWidth.W)
    val y     = (v >> 2) + half                    // ~0.25*v + 0.5
    Mux(y > one, one, Mux(y < zero, zero, y))
  }
  // SInt-friendly overload
  def sigmoid(v: SInt, fracBits: Int): SInt = sigmoidHardS(v, fracBits)
  // UInt wrapper
  def sigmoid(v: UInt, fracBits: Int): UInt = sigmoidHardS(v.asSInt, fracBits).asUInt
}
