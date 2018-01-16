package la_test

import (
	. "github.com/onsi/ginkgo"
	. "github.com/onsi/gomega"

	"testing"
)

func TestLa(t *testing.T) {
	RegisterFailHandler(Fail)
	RunSpecs(t, "La Suite")
}
