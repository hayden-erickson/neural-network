package la

import (
	"math"
	"math/rand"
	"time"
)

func gaussian() float64 {
	val := rand.Float64()

	if rand.Intn(2) == 1 {
		val = -val
	}

	return val
}

func RandVector(n int) []float64 {
	rand.Seed(time.Now().UnixNano())

	var v []float64

	for i := 0; i < n; i++ {
		v = append(v, gaussian())
	}

	return v
}

func RandMatrixSquashed(n, m int) Matrix {
	mat := RandMatrix(n, m)
	return MSCALE(mat, 1/math.Sqrt(float64(m)))
}

func RandMatrix(n, m int) Matrix {
	var d []float64

	for i := 0; i < n; i++ {
		d = append(d, RandVector(m)...)
	}

	return matrix{
		x:    n,
		y:    m,
		data: d,
	}
}

func ZeroMatrix(n, m int) Matrix {
	return matrix{
		x:    n,
		y:    m,
		data: make([]float64, n*m),
	}
}

func NewColMatrix(n, m int, d []float64) Matrix {
	return colmajmatrix{
		x:    n,
		y:    m,
		data: d,
	}
}

func NewMatrix(data [][]float64, colmaj bool) Matrix {
	var n, m int
	n = len(data)
	m = len(data[0])

	d := make([]float64, n*m)

	for i := 0; i < n; i++ {
		for j := 0; j < m; j++ {
			d[i*m+j] = data[i][j]
		}
	}

	if colmaj {
		return colmajmatrix{
			x:    m,
			y:    n,
			data: d,
		}
	}

	return matrix{
		x:    n,
		y:    m,
		data: d,
	}
}

func Dot(a, b []float64) float64 {
	return AddReduce(VMULT(a, b))
}

func Outer(a, b []float64) Matrix {
	var d []float64

	for i, _ := range a {
		d = append(d, Map(b, MultBy(a[i]))...)
	}

	return matrix{
		x:    len(a),
		y:    len(b),
		data: d,
	}
}
