package la

import "errors"

type Array interface {
	Shape() []int
}

type Scalar float64
type Vector []Scalar
type Matrix []Vector

type SOP func(Scalar, Scalar) Scalar

func (v Vector) Shape() []int {
	return []int{len(v)}
}

func (m Matrix) Shape() []int {
	return []int{len(m), len(m[0])}
}

func OP(a, b Vector, op SOP) (Vector, error) {
	if len(a) != len(b) {
		return []Scalar{}, errors.New(`Vectors of unequal size`)
	}

	var c Vector

	for i := 0; i < len(a); i++ {
		c = append(c, op(a[i], b[i]))
	}

	return c, nil
}

func Vdot(a, b Vector) (Scalar, error) {
	var sum Scalar

	if len(a) != len(b) {
		return Scalar(0.0), errors.New(`Vectors of unequal size`)
	}

	for i := 0; i < len(a); i++ {
		sum += a[i] * b[i]
	}

	return sum, nil
}

func MVdot(m Matrix, v Vector) (Vector, error) {
	var c Vector

	for i := 0; i < m.Shape()[0]; i++ {
		s, err := Vdot(m[i], v)

		if err != nil {
			return []Scalar{}, errors.New(`Inner dimension of matrix and vector do not match`)
		}

		c = append(c, s)
	}

	return c, nil
}

func Map(v Vector, apply func(Scalar) Scalar) Vector {
	var newV Vector

	for i := 0; i < len(v); i++ {
		newV = append(newV, apply(v[i]))
	}

	return newV
}
