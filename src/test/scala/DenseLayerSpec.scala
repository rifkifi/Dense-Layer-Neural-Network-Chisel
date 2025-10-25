import chisel3._
import chiseltest._
import org.scalatest.flatspec.AnyFlatSpec

class DenseLayerSpec extends AnyFlatSpec with ChiselScalatestTester {
  behavior of "DenseLayer"

  // Helper: interpret peeked SInt BigInt as signed value with width w
  private def toSigned(b: BigInt, w: Int): Long = {
    val mod = BigInt(1) << w
    val half = BigInt(1) << (w - 1)
    val s = if (b >= half) b - mod else b
    s.longValue
  }

  it should "saturate on overflow (identity activation)" in {
    test(new DenseLayer(x = 2, in_w = 8, n = 2, af = 0, fracBits = 6)) { dut =>
      import Quantization._

      // Inputs: [1.5, 1.5] (Q1.6)
      val inFixed = Seq(1.5, 1.5).map(v => toFixedSInt(v, in_w = 8, fracBits = 6))
      dut.io.in(0).poke(inFixed(0))
      dut.io.in(1).poke(inFixed(1))

      // Weights:
      //  neuron 0: [ +1.5, +1.5 ]
      //  neuron 1: [ -1.5, -1.5 ]
      val wFixed = Seq(
        Seq(1.5, 1.5),
        Seq(-1.5, -1.5)
      ).map(row => row.map(v => toFixedSInt(v, in_w = 8, fracBits = 6)))
      for (j <- 0 until 2; i <- 0 until 2) {
        dut.io.w(j)(i).poke(wFixed(j)(i))
      }

      // Biases: [ +1.0, -1.0 ]
      val bFixed = Seq(1.0, -1.0).map(v => toFixedSInt(v, in_w = 8, fracBits = 6))
      dut.io.b(0).poke(bFixed(0))
      dut.io.b(1).poke(bFixed(1))

      dut.clock.step()

      // Expect saturation to signed 8-bit: [+127, -128]
      val o0 = toSigned(dut.io.out(0).peek().litValue, 8)
      val o1 = toSigned(dut.io.out(1).peek().litValue, 8)
      println(s"Identity overflow -> out0=$o0 expect=127, out1=$o1 expect=-128")
      dut.io.out(0).expect(127.S(8.W))
      dut.io.out(1).expect((-128).S(8.W))
    }
  }

  it should "apply tanh and keep outputs within [-1,1]" in {
    test(new DenseLayer(x = 2, in_w = 8, n = 1, af = 2, fracBits = 6)) { dut =>
      import Quantization._

      // Inputs and weights set to produce a large positive sum
      dut.io.in(0).poke(toFixedSInt(2.0, in_w = 8, fracBits = 6))
      dut.io.in(1).poke(toFixedSInt(2.0, in_w = 8, fracBits = 6))
      dut.io.w(0)(0).poke(toFixedSInt(2.0, in_w = 8, fracBits = 6))
      dut.io.w(0)(1).poke(toFixedSInt(2.0, in_w = 8, fracBits = 6))
      dut.io.b(0).poke(toFixedSInt(0.0, in_w = 8, fracBits = 6))

      dut.clock.step()

      // tanh hard clamps to +1.0 => 64 in Q1.6, within 8-bit
      val ot = toSigned(dut.io.out(0).peek().litValue, 8)
      println(s"Tanh clamp -> out0=$ot expect=64 (Q1.6)")
      dut.io.out(0).expect(64.S(8.W))
    }
  }

  it should "apply ReLU and zero-out negatives" in {
    test(new DenseLayer(x = 2, in_w = 8, n = 2, af = 1, fracBits = 6)) { dut =>
      import Quantization._

      // in = [0.5, -0.75]
      dut.io.in(0).poke(toFixedSInt(0.5, in_w = 8, fracBits = 6))
      dut.io.in(1).poke(toFixedSInt(-0.75, in_w = 8, fracBits = 6))

      // neuron0 weights = [1.0, 1.0] -> sum negative -> ReLU = 0
      // neuron1 weights = [1.0, -0.5] -> sum positive -> ReLU passes through
      dut.io.w(0)(0).poke(toFixedSInt(1.0, in_w = 8, fracBits = 6))
      dut.io.w(0)(1).poke(toFixedSInt(1.0, in_w = 8, fracBits = 6))
      dut.io.w(1)(0).poke(toFixedSInt(1.0, in_w = 8, fracBits = 6))
      dut.io.w(1)(1).poke(toFixedSInt(-0.5, in_w = 8, fracBits = 6))

      // biases = [0.0, 0.0]
      dut.io.b(0).poke(toFixedSInt(0.0, in_w = 8, fracBits = 6))
      dut.io.b(1).poke(toFixedSInt(0.0, in_w = 8, fracBits = 6))

      dut.clock.step()

      // Expected: neuron0 -> 0, neuron1 -> 56 (Q1.6)
      val r0 = toSigned(dut.io.out(0).peek().litValue, 8)
      val r1 = toSigned(dut.io.out(1).peek().litValue, 8)
      println(s"ReLU -> out0=$r0 expect=0, out1=$r1 expect=56")
      dut.io.out(0).expect(0.S(8.W))
      dut.io.out(1).expect(56.S(8.W))
    }
  }
}
