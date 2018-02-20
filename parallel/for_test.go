package parallel_test

import (
	"testing"

	"github.com/hayden-erickson/neural-network/la"
	. "github.com/hayden-erickson/neural-network/parallel"
)

type tester struct {
	data []float64
	op   la.OP
}

const N = 1 << 25

var t = tester{data: la.RandVector(N), op: la.MultBy(5.0)}

func (t tester) getSerialTime() []float64 {
	out := make([]float64, len(t.data))

	for i := 0; i < len(t.data); i++ {
		out[i] = t.op(t.data[i])
	}

	return out
}

func (t tester) getParallelTime(branch int) []float64 {
	return For(t.data, branch, t.op)
}

func BenchmarkSerial(b *testing.B) {
	for i := 0; i < b.N; i++ {
		t.getSerialTime()
	}
}

func BenchmarkFor2(b *testing.B) {
	for i := 0; i < b.N; i++ {
		t.getParallelTime(1)
	}
}

func BenchmarkFor4(b *testing.B) {
	for i := 0; i < b.N; i++ {
		t.getParallelTime(2)
	}
}

func BenchmarkFor8(b *testing.B) {
	for i := 0; i < b.N; i++ {
		t.getParallelTime(3)
	}
}

func BenchmarkFor16(b *testing.B) {
	for i := 0; i < b.N; i++ {
		t.getParallelTime(4)
	}
}

// var N int
// var i uint

// fmt.Println("# N\t0\t2\t4\t8\t16")
// pTime := make([]int64, 4)

// for i = 10; i < 32; i++ {
// 	N = 1 << i

// 	sTime := t.getSerialTime()
// 	pTime[0] = t.getParallelTime(1)
// 	pTime[1] = t.getParallelTime(2)
// 	pTime[2] = t.getParallelTime(3)
// 	pTime[3] = t.getParallelTime(4)

// 	printRow(N, sTime, pTime)
// }
