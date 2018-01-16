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

func RandVector(n int) Vector {
	rand.Seed(time.Now().UnixNano())

	var v Vector

	for i := 0; i < n; i++ {
		v = append(v, Scalar(gaussian()))
	}

	return v
}

func RandMatrix(n, m int) Matrix {
	var mat Matrix

	for i := 0; i < n; i++ {
		mat = append(mat, RandVector(m))
	}

	return mat
}

func ZeroMatrix(n, m int) Matrix {
	var mat Matrix

	for i := 0; i < n; i++ {
		mat = append(mat, make(Vector, m))
	}

	return mat
}
