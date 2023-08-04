// 
// Adapted from three.js
// license: MIT (https://github.com/mrdoob/three.js/blob/dev/LICENSE)
// url: https://github.com/mrdoob/three.js/blob/dev/src/math/Sphere.js
//

import {Vector3} from "./Vector3.js";

export class Sphere {

	constructor( center = new Vector3(), radius = - 1 ) {

		this.center = center;
		this.radius = radius;

	}

	set( center, radius ) {

		this.center.copy( center );
		this.radius = radius;

		return this;

	}

	fromBox(box){
		this.center.x = 0.5 * (box.min.x + box.max.x);
		this.center.y = 0.5 * (box.min.y + box.max.y);
		this.center.z = 0.5 * (box.min.z + box.max.z);

		let dx = box.max.x - box.min.x;
		let dy = box.max.y - box.min.y;
		let dz = box.max.z - box.min.z;
		
		this.radius = 0.5 * Math.sqrt(dx * dx + dy * dy + dz * dz);
	}

	copy( sphere ) {

		this.center.copy( sphere.center );
		this.radius = sphere.radius;

		return this;

	}

	containsPoint( point ) {

		return ( point.distanceToSquared( this.center ) <= ( this.radius * this.radius ) );

	}

	distanceToPoint( point ) {

		return ( point.distanceTo( this.center ) - this.radius );

	}

	intersectsSphere( sphere ) {

		const radiusSum = this.radius + sphere.radius;

		return sphere.center.distanceToSquared( this.center ) <= ( radiusSum * radiusSum );

	}

	applyMatrix4( matrix ) {

		this.center.applyMatrix4( matrix );
		this.radius = this.radius * matrix.getMaxScaleOnAxis();

		return this;

	}

}
