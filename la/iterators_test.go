package la_test

import (
	"testing"

	. "github.com/hayden-erickson/neural-network/la"
)

func NOOP(z float64) float64             { return z }
func NOBOP(a, b float64) float64         { return a }
func NOOPI(z float64, is ...int) float64 { return z }

func BenchmarkMMap(b *testing.B) {
	m := RandMatrix(50, 100)

	for i := 0; i < b.N; i++ {
		MMap(m, NOOP)
	}
}

func BenchmarkMMapD(b *testing.B) {
	m := RandMatrix(50, 100)

	for i := 0; i < b.N; i++ {
		MMapD(m, NOOP)
	}
}

func BenchmarkMMapI(b *testing.B) {
	m := RandMatrix(50, 100)

	for i := 0; i < b.N; i++ {
		MMapI(m, NOOPI)
	}
}

func BenchmarkMMapID(b *testing.B) {
	m := RandMatrix(50, 100)

	for i := 0; i < b.N; i++ {
		MMapID(m, NOOPI)
	}
}

func BenchmarkMAgg(b *testing.B) {
	a := RandMatrix(500, 100)
	c := RandMatrix(500, 100)

	for i := 0; i < b.N; i++ {
		MAgg(a, c, NOBOP)
	}

}

func BenchmarkMAggD(b *testing.B) {
	a := RandMatrix(500, 100)
	c := RandMatrix(500, 100)

	for i := 0; i < b.N; i++ {
		MAggD(a, c, NOBOP)
	}

}
