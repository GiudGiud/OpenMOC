#include "Vector.h"

/**
 * @brief Constructor initializes Vector object as a floating point array
 *        and sets the vector dimensions.
 * @detail The vector is ordered by cell (as opposed to by group) on the
 *         outside to be consistent with the Matrix object. Locks are used to
 *         make the vector object thread-safe against concurrent writes the 
 *          same value. One lock locks out multiple rows of
 *         the vector at a time representing multiple groups in the same cell.
 * @param num_x The number of cells in the x direction.
 * @param num_y The number of cells in the y direction.
 * @param num_groups The number of energy groups in each cell.
 */
Vector::Vector(int num_x, int num_y, int num_groups) {

  setNumX(num_x);
  setNumY(num_y);
  setNumGroups(num_groups);  
  _num_rows = _num_x*_num_y*_num_groups;

  /* Initialize array and set all to 0.0 */
  _array = new FP_PRECISION[_num_rows];
  setAll(0.0);

  /* Allocate memory for OpenMP locks for each Vector cell */ 
  _cell_locks = new omp_lock_t[_num_x*_num_y];

  /* Loop over all Vector cells to initialize OpenMP locks */
  #pragma omp parallel for schedule(guided)
  for (int r=0; r < _num_x*_num_y; r++)
    omp_init_lock(&_cell_locks[r]);
}


/**
 * @brief Destructor deletes the arrays used to represent the vector.
 */
Vector::~Vector() {

  if (_array != NULL)
    delete [] _array;

  if (_cell_locks != NULL)
    delete [] _cell_locks;  
}


/**
 * @brief Increment a value in the vector.
 * @detail This method takes a cell and group and floating
 *         point value. The cell and group are used to compute the
 *         row and column in the vector. If a value exists for the row,
 *         the value is incremented by val; otherwise, it is set to val.
 * @param cell The cell location.
 * @param group The group location.
 * @param val The value used to increment the row location.
 */
void Vector::incrementValue(int cell, int group, FP_PRECISION val) {

  if (cell >= _num_x*_num_y || cell < 0)
    log_printf(ERROR, "Unable to increment Vector value for cell %d"
               " which is not between 0 and %d", cell, _num_x*_num_y-1);
  else if (group >= _num_groups || group < 0)
    log_printf(ERROR, "Unable to increment Vector value for group %d"
               " which is not between 0 and %d", group, _num_groups-1);

  /* Atomically increment the Vector value from the
   * temporary array using mutual exclusion locks */
  omp_set_lock(&_cell_locks[cell]);

  _array[cell*_num_groups + group] += val;

  /* Release Vector cell mutual exclusion lock */
  omp_unset_lock(&_cell_locks[cell]);
}


void Vector::setAll(FP_PRECISION val) {
  std::fill_n(_array, _num_rows, val);
}


/**
 * @brief Set a value in the vector.
 * @detail This method takes a cell and group and floating
 *         point value. The cell and group are used to compute the
 *         row and column in the vector. The location of the corresponding
 *         row is set to val.
 * @param cell The cell location.
 * @param group The group location.
 * @param val The value used to set the row location.
 */
void Vector::setValue(int cell, int group, FP_PRECISION val) {

  if (cell >= _num_x*_num_y || cell < 0)
    log_printf(ERROR, "Unable to set Vector value for cell %d"
               " which is not between 0 and %d", cell, _num_x*_num_y-1);
  else if (group >= _num_groups || group < 0)
    log_printf(ERROR, "Unable to set Vector value for group %d"
               " which is not between 0 and %d", group, _num_groups-1);

  /* Atomically set the Vector value from the
   * temporary array using mutual exclusion locks */
  omp_set_lock(&_cell_locks[cell]);

  _array[cell*_num_groups + group] = val;

  /* Release Vector cell mutual exclusion lock */
  omp_unset_lock(&_cell_locks[cell]);
}


/**
 * @brief Clear all values in the vector.
 */
void Vector::clear() {
  setAll(0.0);
}


/**
 * @brief Scales the vector by a given value.
 * @param val The value to scale the vector by.
 */
void Vector::scaleByValue(FP_PRECISION val) {

  #pragma omp parallel for schedule(guided)
  for (int i=0; i < _num_rows; i++)
    _array[i] *= val;
}


/**
 * @brief Print the vector object to the log file.
 */
void Vector::printString() {

  std::stringstream string;
  string << std::setprecision(6);

  string << std::endl;
  string << "Vector" << std::endl;
  string << " Num rows: " << _num_rows << std::endl;

  for (int row=0; row < _num_rows; row++)
    string << " ( " << row << "): " << _array[row] << std::endl;

  string << "End Vector" << std::endl;

  log_printf(NORMAL, string.str().c_str());
}


/**
 * @brief Copy the values from the current vector to an input vector.
 * @param vector The vector to copy values to.
 */
void Vector::copyTo(Vector* vector) {
  std::copy(_array, _array + _num_rows, vector->getArray());
}


/**
 * @brief Get a value at location described by a given cell and 
 *        group index.
 * @param cell The cell location index.
 * @param group The group location index.
 */
FP_PRECISION Vector::getValue(int cell, int group) {
  return _array[cell*_num_groups + group];
}


/**
 * @brief Get the array describing the vector.
 * @return The array describing the vector.
 */
FP_PRECISION* Vector::getArray() {
  return _array;
}


/**
 * @brief Get the number of cells in the x dimension.
 * @return The number of cells in the x dimension.
 */
int Vector::getNumX() {
  return _num_x;
}


/**
 * @brief Get the number of cells in the y dimension.
 * @return The number of cells in the y dimension.
 */
int Vector::getNumY() {
  return _num_y;
}


/**
 * @brief Get the number of groups in each cell.
 * @return The number of groups in each cell.
 */
int Vector::getNumGroups() {
  return _num_groups;
}


/**
 * @brief Get the number of rows in the vector.
 * @return The number of rows in the vector.
 */
int Vector::getNumRows() {
  return _num_rows;
}


/**
 * @brief Get the sum of all the values in the vector.
 * @return The sum of all the values in the vector.
 */
FP_PRECISION Vector::getSum() {
  return pairwise_sum(_array, _num_rows);
}


/**
 * @brief Set the number of cells in the x dimension.
 * @param num_x The number of cells in the x dimension.
 */
void Vector::setNumX(int num_x) {

  if (num_x < 1)
    log_printf(ERROR, "Unable to set Vector num x to non-positive value %d",
               num_x);

  _num_x = num_x;
}


/**
 * @brief Set the number of cells in the y dimension.
 * @param num_y The number of cells in the y dimension.
 */
void Vector::setNumY(int num_y) {

  if (num_y < 1)
    log_printf(ERROR, "Unable to set Vector num y to non-positive value %d",
               num_y);

  _num_y = num_y;
}


/**
 * @brief Set the number of groups in each cell.
 * @param num_groups The number of groups in each cell.
 */
void Vector::setNumGroups(int num_groups) {

  if (num_groups < 1)
    log_printf(ERROR, "Unable to set Vector num groups to non-positive value"
               " %d", num_groups);

  _num_groups = num_groups;
}
