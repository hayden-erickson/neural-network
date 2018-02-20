package la

type Matrix interface {
	T() Matrix
	Row(int) []float64
	Col(int) []float64
	At(i, j int) *float64
	Shape() []int
	Data() []float64
}

type matrix struct {
	data []float64
	x    int
	y    int
}

type colmajmatrix struct {
	data []float64
	x    int
	y    int
}

type transposer struct {
	m matrix
}

func (t transposer) T() Matrix {
	return t.m
}

func (t transposer) Row(i int) []float64 {
	return t.m.Col(i)
}

func (t transposer) Col(j int) []float64 {
	return t.m.Row(j)
}

func (t transposer) At(i, j int) *float64 {
	return t.m.At(j, i)
}

func (t transposer) Shape() []int {
	return []int{t.m.y, t.m.x}
}

func (cm colmajmatrix) T() Matrix {
	return matrix{
		x:    cm.y,
		y:    cm.x,
		data: cm.data,
	}
}

func (cm colmajmatrix) Row(j int) []float64 {
	out := make([]float64, cm.y)

	for i := 0; i < cm.y; i++ {
		out[i] = cm.data[i*cm.x+j]
	}

	return out
}

func (cm colmajmatrix) Col(j int) []float64 {
	return cm.data[(j * cm.x):((j + 1) * cm.x)]
}

func (cm colmajmatrix) At(i int, j int) *float64 {
	return &cm.data[j*cm.x+i]
}

func (cm colmajmatrix) Shape() []int {
	return []int{cm.x, cm.y}
}

// return the data in row major order
// for compatibility with data based
// iterators using 1-dimensional
// array indexing for speed
func (cm colmajmatrix) Data() []float64 {
	out := make([]float64, cm.x*cm.y)

	for i := 0; i < cm.x*cm.y; i++ {
		I := i / cm.y
		J := i % cm.y
		out[I*cm.y+J] = cm.data[i]
	}

	return out
}

func (t transposer) Data() []float64 {
	out := make([]float64, t.m.x*t.m.y)

	for i := 0; i < t.m.x*t.m.y; i++ {
		I := i / t.m.y
		J := i % t.m.y
		out[i] = t.m.data[J*t.m.y+I]
	}

	return out
}

func (m matrix) T() Matrix {
	return transposer{m: m}
}

func (m matrix) Row(i int) []float64 {
	return m.data[(i * m.y):((i + 1) * m.y)]
}

func (m matrix) Col(j int) []float64 {
	out := make([]float64, m.x)

	for i := 0; i < m.x; i++ {
		out[i] = m.data[i*m.y+j]
	}

	return out
}

func (m matrix) At(i, j int) *float64 {
	return &m.data[i*m.y+j]
}

func (m matrix) Shape() []int {
	return []int{m.x, m.y}
}

func (m matrix) Data() []float64 {
	return m.data
}

// func CreateMatrixAgg
