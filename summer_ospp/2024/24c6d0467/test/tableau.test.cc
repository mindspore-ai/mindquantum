#include <gtest/gtest.h>
#include "tableau/tableau.hpp"

TEST(tableau_test, x_test) {
    Tableau tableau(1);
    tableau.do_X(0);
}