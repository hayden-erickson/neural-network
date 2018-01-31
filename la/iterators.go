package la

type OP func(a float64) float64
type BOP func(a, b float64) float64
type IOP func(a float64, is ...int) float64

type Reducer func([]float64) float64
type MReducer func([]Matrix) Matrix
type Mapper func([]float64) []float64
type VectorBOP func(a, b []float64) []float64
type MatrixBOP func(a, b Matrix) Matrix
type MatrixMapper func(Matrix) Matrix

func Map(a []float64, op OP) []float64 {
	out := make([]float64, len(a))

	for i, _ := range a {
		out[i] = op(a[i])
	}

	return out
}

func MapVectorCol(a []float64, op BOP) IOP {
	return func(b float64, is ...int) float64 {
		return op(b, a[is[0]])
	}
}

func Agg(a, b []float64, op BOP) []float64 {
	out := make([]float64, len(a))

	for i, _ := range a {
		out[i] = op(a[i], b[i])
	}

	return out
}

func MAgg(a, b Matrix, op BOP) Matrix {
	m := matrix{x: a.Shape()[0], y: b.Shape()[1]}
	m.data = make([]float64, a.Shape()[0]*a.Shape()[1])

	for i := 0; i < a.Shape()[0]; i++ {
		for j := 0; j < a.Shape()[1]; j++ {
			*m.At(i, j) = op(*a.At(i, j), *b.At(i, j))
		}
	}

	return m
}

func Reduce(a []float64, op BOP) float64 {
	out := a[0]

	for i := 1; i < len(a); i++ {
		out = op(out, a[i])
	}

	return out
}

func MReduce(a []Matrix, op BOP) Matrix {
	out := a[0]

	for i := 1; i < len(a); i++ {
		out = MAgg(out, a[i], op)
	}

	return out
}

func MMapI(m Matrix, op IOP) Matrix {
	for i := 0; i < m.Shape()[0]; i++ {
		for j := 0; j < m.Shape()[1]; j++ {
			*m.At(i, j) = op(*m.At(i, j), i, j)
		}
	}

	return m
}

func MMap(m Matrix, op OP) Matrix {
	for i := 0; i < m.Shape()[0]; i++ {
		for j := 0; j < m.Shape()[1]; j++ {
			*m.At(i, j) = op(*m.At(i, j))
		}
	}

	return m
}

func MVDot(a Matrix, b []float64) []float64 {
	var d []float64

	for i := 0; i < a.Shape()[0]; i++ {
		d = append(d, Dot(a.Row(i), b))
	}

	return d
}

func MMDot(a, b Matrix) Matrix {
	out := ZeroMatrix(a.Shape()[0], b.Shape()[1])

	for i := 0; i < a.Shape()[0]; i++ {
		for j := 0; j < b.Shape()[1]; j++ {
			*out.At(i, j) = Dot(a.Row(i), b.Col(j))
		}
	}

	return out
}

func CreateMReducer(op BOP) MReducer {
	return func(a []Matrix) Matrix {
		return MReduce(a, op)
	}
}

func CreateVectorOP(op BOP) VectorBOP {
	return func(a, b []float64) []float64 {
		return Agg(a, b, op)
	}
}

func CreateVMapper(op OP) Mapper {
	return func(a []float64) []float64 {
		return Map(a, op)
	}
}

func CreateMatrixOP(op BOP) MatrixBOP {
	return func(a, b Matrix) Matrix {
		return MAgg(a, b, op)
	}
}

func CreateReducer(op BOP) Reducer {
	return func(a []float64) float64 {
		return Reduce(a, op)
	}
}

func CreateMMapper(op OP) MatrixMapper {
	return func(m Matrix) Matrix {
		return MMap(m, op)
	}
}

func Add(x float64) OP {
	return func(a float64) float64 {
		return a + x
	}
}

func MultBy(b float64) OP {
	return func(a float64) float64 {
		return a * b
	}
}

func SUM(a, b float64) float64 {
	return a + b
}

func SUB(a, b float64) float64 {
	return a - b
}

func MULT(a, b float64) float64 {
	return a * b
}

func VSCALE(a []float64, x float64) []float64 {
	scale := CreateVMapper(MultBy(x))
	return scale(a)
}

func MSCALE(a Matrix, x float64) Matrix {
	scale := CreateMMapper(MultBy(x))
	return scale(a)
}

// vector operations
var VSUM = CreateVectorOP(SUM)
var VSUB = CreateVectorOP(SUB)
var VMULT = CreateVectorOP(MULT)
var AddReduce = CreateReducer(SUM)

// matrix operations
var MSUM = CreateMatrixOP(SUM)
var MMULT = CreateMatrixOP(MULT)
var MAddReduce = CreateMReducer(SUM)
