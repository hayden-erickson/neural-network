package la

import (
	"github.com/hayden-erickson/neural-network/parallel"
)

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

// app the given vector to each column of a matrix
func MapVectorCol(a []float64, op BOP) IOP {
	return func(b float64, is ...int) float64 {
		return op(b, a[is[0]])
	}
}

// a binary operation for two vectors
func Agg(a, b []float64, op BOP) []float64 {
	out := make([]float64, len(a))

	for i, _ := range a {
		out[i] = op(a[i], b[i])
	}

	return out
}

// a binary operation for two matricies
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

// an optimized binary operation for two matricies
// D means data. (i.e. use the matricies raw data
// instead of it's 2-D index function At)
// this method yields much better performance which can be seen in the
// corresponding benchmarks
func MAggD(a, b Matrix, op BOP) Matrix {
	out := matrix{x: a.Shape()[0], y: b.Shape()[1]}
	out.data = make([]float64, a.Shape()[0]*a.Shape()[1])

	aData := a.Data()
	bData := b.Data()

	for i := 0; i < len(aData); i++ {
		out.data[i] = op(aData[i], bData[i])
	}

	return out
}

func Reduce(a []float64, op BOP) float64 {
	out := a[0]

	for i := 1; i < len(a); i++ {
		out = op(out, a[i])
	}

	return out
}

// reduce a list of matricies to a single matrix
// all matricies must have the same shape
func MReduce(a []Matrix, op BOP) Matrix {
	out := a[0]

	for i := 1; i < len(a); i++ {
		out = MAggD(out, a[i], op)
	}

	return out
}

// M = matrix, Map, I = indexed
// a special mapper which passes along
// the current index to the operator
func MMapI(m Matrix, op IOP) Matrix {
	for i := 0; i < m.Shape()[0]; i++ {
		for j := 0; j < m.Shape()[1]; j++ {
			*m.At(i, j) = op(*m.At(i, j), i, j)
		}
	}

	return m
}

// M = matrix, Map, I = indexed, D = data
// an optimized form of the above function
// performance increases can be seen in the
// corresponding benchmarks
func MMapID(m Matrix, op IOP) Matrix {
	data := m.Data()

	out := matrix{x: m.Shape()[0], y: m.Shape()[1]}
	out.data = make([]float64, m.Shape()[0]*m.Shape()[1])

	numCols := m.Shape()[1]

	for i := 0; i < len(data); i++ {
		I := i / numCols
		J := i % numCols
		out.data[i] = op(data[i], I, J)
	}

	return out
}

// M = matrix, Map
// apply the unary operator element wise to
// the given matrix
func MMap(m Matrix, op OP) Matrix {
	x := m.Shape()[0]
	y := m.Shape()[1]

	for i := 0; i < x; i++ {
		for j := 0; j < y; j++ {
			*m.At(i, j) = op(*m.At(i, j))
		}
	}

	return m
}

// M = matrix, Map, D = data
// an optimized form of the above function
func MMapD(m Matrix, op OP) Matrix {
	data := m.Data()
	out := make([]float64, len(data))

	for i := 0; i < len(data); i++ {
		out[i] = op(data[i])
	}

	return matrix{
		x:    m.Shape()[0],
		y:    m.Shape()[1],
		data: out,
	}
}

// M = matrix, Map, D = data, P = parallel
// use the par for loop with a branch factor
// of two (i.e. 2^2 = 4 threads)
func MMapDP(m Matrix, op OP) Matrix {
	parallel.For(m.Data(), 2, op)
	return m
}

// M = matrix, V = vector, Dot product
func MVDot(a Matrix, b []float64) []float64 {
	var d []float64

	for i := 0; i < a.Shape()[0]; i++ {
		d = append(d, Dot(a.Row(i), b))
	}

	return d
}

// M = matrix, M = matrix, Dot product
func MMDot(a, b Matrix) Matrix {
	out := ZeroMatrix(a.Shape()[0], b.Shape()[1])

	for i := 0; i < a.Shape()[0]; i++ {
		for j := 0; j < b.Shape()[1]; j++ {
			*out.At(i, j) = Dot(a.Row(i), b.Col(j))
		}
	}

	return out
}

// === Helper iterator functions ===
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
		return MAggD(a, b, op)
	}
}

func CreateReducer(op BOP) Reducer {
	return func(a []float64) float64 {
		return Reduce(a, op)
	}
}

func CreateMMapper(op OP) MatrixMapper {
	return func(m Matrix) Matrix {
		return MMapD(m, op)
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

func RowAvg(a Matrix) []float64 {
	out := make([]float64, a.Shape()[0])

	for i := 0; i < a.Shape()[0]; i++ {
		out[i] = AddReduce(a.Row(i))
	}

	return VSCALE(out, (1 / float64(a.Shape()[1])))
}

func MOuterColAvg(a, b Matrix) Matrix {
	out := make([]Matrix, a.Shape()[1])

	for j := 0; j < a.Shape()[1]; j++ {
		out[j] = Outer(a.Col(j), b.Col(j))
	}

	return MSCALE(MAddReduce(out), (1 / float64(a.Shape()[1])))

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
