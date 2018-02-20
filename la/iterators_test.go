package la_test

import (
	"testing"

	. "github.com/hayden-erickson/neural-network/la"
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"
)

func NOOP(z float64) float64             { return z }
func NOBOP(a, b float64) float64         { return a }
func NOOPI(z float64, is ...int) float64 { return z }

var m = RandMatrix(5000, 100)
var a = RandMatrix(500, 100)
var c = RandMatrix(500, 100)

var _ = Describe("Iterators", func() {
	Describe("#Reduce", func() {
		It("reduces the set to a single value using the given operator", func() {
			Expect(Reduce([]float64{1, 2, 3, 4}, SUM)).
				To(BeEquivalentTo(10))
		})
	})

	Describe("#MReduce", func() {
		It("adds up all matricies into one", func() {
			a := NewMatrix([][]float64{
				{1, 2, 3},
				{4, 5, 6},
			}, false)

			b := NewMatrix([][]float64{
				{6, 5, 4},
				{3, 2, 1},
			}, false)

			c := NewMatrix([][]float64{
				{3, 3, 3},
				{3, 3, 3},
			}, false)

			Expect(NewMatrix([][]float64{
				{10, 10, 10},
				{10, 10, 10},
			}, false)).To(Equal(
				MReduce([]Matrix{a, b, c}, SUM)))
		})
	})

	Describe("#RowAvg", func() {
		It("reduces the matrix row wise into a vector and averages each vector element", func() {

			m := NewMatrix([][]float64{
				{1, 2, 3, 4},
				{5, 6, 7, 8},
			}, false)

			Expect([]float64{(10 / 4.0), (26 / 4.0)}).To(Equal(RowAvg(m)))
		})
	})

	Describe("#MOuterColAvg", func() {
		It("takes the column-wise outer product of both matricies and reduces then averages each element in the resulting matrix", func() {
			m := NewMatrix([][]float64{
				{2, 6},
				{4, 8},
			}, false)

			n := NewMatrix([][]float64{
				{1, 2},
				{1, 2},
				{1, 2},
				{1, 2},
			}, false)

			// 2, 2, 2, 2
			// 4, 4, 4, 4
			// +
			// 12, 12, 12, 12
			// 16, 16, 16, 16
			// =
			// 14, 14, 14, 14
			// 20, 20, 20, 20
			// avg = x/2 =
			// 7, 7, 7, 7
			// 10, 10, 10, 10

			Expect(NewMatrix([][]float64{
				{7, 7, 7, 7},
				{10, 10, 10, 10},
			}, false)).To(Equal(MOuterColAvg(m, n)))
		})

	})
})

func BenchmarkMMap(b *testing.B) {
	for i := 0; i < b.N; i++ {
		MMap(m, NOOP)
	}
}

func BenchmarkMMapD(b *testing.B) {
	for i := 0; i < b.N; i++ {
		MMapD(m, NOOP)
	}
}

func BenchmarkMMapDP(b *testing.B) {
	for i := 0; i < b.N; i++ {
		MMapDP(m, NOOP)
	}
}

func BenchmarkMMapI(b *testing.B) {
	for i := 0; i < b.N; i++ {
		MMapI(m, NOOPI)
	}
}

func BenchmarkMMapID(b *testing.B) {
	for i := 0; i < b.N; i++ {
		MMapID(m, NOOPI)
	}
}

func BenchmarkMAgg(b *testing.B) {
	for i := 0; i < b.N; i++ {
		MAgg(a, c, NOBOP)
	}

}

func BenchmarkMAggD(b *testing.B) {
	for i := 0; i < b.N; i++ {
		MAggD(a, c, NOBOP)
	}

}
