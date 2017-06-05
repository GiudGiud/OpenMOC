/**
 * @file Region.h
 * @brief The Region class.
 * @date March 10, 2017
 * @author William Boyd, MIT, Course 22 (wboyd@mit.edu)
 */


#ifndef REGION_H_
#define REGION_H_

#ifdef __cplusplus
#ifdef SWIG
#include "Python.h"
#endif
#include "Surface.h"
#include "boundary_type.h"
#include <limits>
#endif


/* Forward declarations to resolve circular dependencies */
class Intersection;
class Union;
class Complement;
class Halfspace;


/**
 * @enum regionType
 * @brief The types of regions supported by OpenMOC.
 */
enum regionType {
  /** The intersection of one or more regions */
  INTERSECTION,

  /** The union of one or more regions */
  UNION,

  /** The complement of a region */
  COMPLEMENT,

  /** The side of a surface */
  HALFSPACE

};


/**
 * @class Region Region.h "src/Region.h"
 * @brief A region of space that can be assigned to a Cell.
 */
class Region {

protected:

  /** The type of Region (ie, UNION, INTERSECTION, etc) */
  regionType _region_type;

  /** A collection of the nodes within the Region */
  std::vector<Region*> _nodes;

public:
  virtual ~Region();
  virtual void addNode(Region* node);
  virtual std::vector<Region*> getNodes();
  virtual std::map<int, Halfspace*> getAllSurfaces();
  regionType getRegionType();

  virtual double getMinX();
  virtual double getMaxX();
  virtual double getMinY();
  virtual double getMaxY();
  virtual double getMinZ();
  virtual double getMaxZ();
  virtual boundaryType getMinXBoundaryType();
  virtual boundaryType getMaxXBoundaryType();
  virtual boundaryType getMinYBoundaryType();
  virtual boundaryType getMaxYBoundaryType();

  virtual bool containsPoint(Point* point) =0;
  virtual double minSurfaceDist(LocalCoords* coords);
  virtual Region* clone();
};


/**
 * @class Intersection Intersection.h "src/Region.h"
 * @brief An intersection of two or more Regions.
 */
class Intersection : public Region {

public:
  Intersection();
  bool containsPoint(Point* point);
};


/**
 * @class Union Union.h "src/Region.h"
 * @brief A union of two or more Regions.
 */
class Union : public Region {

 public:
  Union();
  bool containsPoint(Point* point);
};


/**
 * @class Complement Complement.h "src/Region.h"
 * @brief A complement of a Region.
 */
class Complement : public Region {

public:
  Complement();
  bool containsPoint(Point* point);  
};



/**
 * @class Halfspace Halfspace.h "src/Region.h"
 * @brief A positive or negative halfspace Region.
 */
class Halfspace : public Region {

private:

  /** A pointer to the Surface object */
  Surface* _surface;

  /** The halfspace associated with this surface */
  int _halfspace;
  
public:
  Halfspace(int halfspace, Surface* surface);
  Halfspace* clone();

  Surface* getSurface();
  int getHalfspace();
  std::map<int, Halfspace*> getAllSurfaces();

  double getMinX();
  double getMaxX();
  double getMinY();
  double getMaxY();
  double getMinZ();
  double getMaxZ();
  boundaryType getMinXBoundaryType();
  boundaryType getMaxXBoundaryType();
  boundaryType getMinYBoundaryType();
  boundaryType getMaxYBoundaryType();

  bool containsPoint(Point* point);  
  double minSurfaceDist(LocalCoords* coords);
};


/**
 * @class Halfspace Halfspace.h "src/Region.h"
 * @brief A positive or negative halfspace Region.
 */
class RectangularPrism : public Intersection {

public:
  RectangularPrism(double width_x, double width_y,
		   double origin_x=0., double origin_y=0.);
  void setBoundaryType(boundaryType boundary_type);
};

#endif /* REGION_H_ */
