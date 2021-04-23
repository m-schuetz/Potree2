// 
// Adapted from three.js
// license: MIT (https://github.com/mrdoob/three.js/blob/dev/LICENSE)
// url: https://github.com/mrdoob/three.js/blob/dev/src/math/Ray.js
//

import {Vector3} from "./Vector3.js";

const _vector = new Vector3();
const _segCenter = new Vector3();
const _segDir = new Vector3();
const _diff = new Vector3();

const _edge1 = new Vector3();
const _edge2 = new Vector3();
const _normal = new Vector3();

export class Ray{

	constructor(origin, direction){
		this.origin = origin ?? new Vector3();
		this.direction = direction ?? new Vector3( 0, 0, - 1 );
	}

	distanceToPoint( point ) {
		return Math.sqrt( this.distanceSqToPoint( point ) );
	}

	distanceSqToPoint( point ) {

		const directionDistance = _vector.subVectors( point, this.origin ).dot( this.direction );

		// point behind the ray
		if ( directionDistance < 0 ) {

			return this.origin.distanceToSquared( point );

		}

		_vector.copy( this.direction ).multiplyScalar( directionDistance ).add( this.origin );

		return _vector.distanceToSquared( point );

	}

	closestPointToPoint( point ) {

		let target = new Vector3();

		target.subVectors( point, this.origin );

		const directionDistance = target.dot( this.direction );

		if ( directionDistance < 0 ) {

			return target.copy( this.origin );

		}

		return target.copy( this.direction ).multiplyScalar( directionDistance ).add( this.origin );

	}

}

