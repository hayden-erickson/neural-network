package la

import (
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

func NewMatrix(data [][]float64) Matrix {
	n := len(data)
	m := len(data[0])

	d := make([]float64, n*m)

	for i, _ := range data {
		for j, _ := range data[i] {
			d[i*m+j] = data[i][j]
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
