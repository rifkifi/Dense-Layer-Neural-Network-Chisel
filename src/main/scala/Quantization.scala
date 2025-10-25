import chisel3._

/*
  Fixed-point quantization helpers for DenseLayer inputs, weights, and biases.

  Conventions
  - Values are interpreted as real numbers (Double/Int) and converted to fixed-point SInt.
  - Scale factor is 2^fracBits, so 1.0 == (1 << fracBits).
  - Results are saturated to the target signed width `in_w`.
*/

object Quantization {
  private def clampToWidth(value: Long, in_w: Int): Long = {
    val min = -(1L << (in_w - 1))
    val max = (1L << (in_w - 1)) - 1L
    Math.max(min, Math.min(max, value))
  }

  private def toFixedRaw(value: Double, fracBits: Int): Long = {
    val scale = 1L << fracBits
    Math.round(value * scale)
  }

  def toFixedSInt(value: Double, in_w: Int, fracBits: Int): SInt = {
    val raw = toFixedRaw(value, fracBits)
    val clamped = clampToWidth(raw, in_w)
    BigInt(clamped).S(in_w.W)
  }

  def toFixedSInt(value: Int, in_w: Int, fracBits: Int): SInt = {
    val raw = value.toLong << fracBits
    val clamped = clampToWidth(raw, in_w)
    BigInt(clamped).S(in_w.W)
  }

  def toFixedVec(values: Seq[Double], in_w: Int, fracBits: Int): Vec[SInt] = {
    VecInit(values.map(v => toFixedSInt(v, in_w, fracBits)))
  }

  def toFixedMatrix(values: Seq[Seq[Double]], in_w: Int, fracBits: Int): Vec[Vec[SInt]] = {
    VecInit(values.map(row => VecInit(row.map(v => toFixedSInt(v, in_w, fracBits)))))
  }

  // Optional: convert back to real for debug/printing (elaboration-time only)
  def fromFixed(value: SInt, fracBits: Int): Double = {
    value.litValue.toDouble / (1 << fracBits)
  }
}
