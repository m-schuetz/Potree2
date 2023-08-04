// 
// Adapted from three.js
// license: MIT (https://github.com/mrdoob/three.js/blob/dev/LICENSE)
// url: https://github.com/mrdoob/three.js/blob/dev/src/math/Frustum.js
//

import {Plane} from "./Plane.js";
import {Vector3} from "./Vector3.js";

const _vector = new Vector3();

export class Frustum{

	constructor( p0, p1, p2, p3, p4, p5 ) {

		this.planes = [

			( p0 !== undefined ) ? p0 : new Plane(),
			( p1 !== undefined ) ? p1 : new Plane(),
			( p2 !== undefined ) ? p2 : new Plane(),
			( p3 !== undefined ) ? p3 : new Plane(),
			( p4 !== undefined ) ? p4 : new Plane(),
			( p5 !== undefined ) ? p5 : new Plane()

		];
	}

	setFromMatrix(m){

		let planes = this.planes;
		let me   = m.elements;
		let me0  = me[ 0 ] , me1  = me[ 1 ] ,  me2 = me[ 2 ] ,  me3 = me[ 3 ];
		let me4  = me[ 4 ] , me5  = me[ 5 ] ,  me6 = me[ 6 ] ,  me7 = me[ 7 ];
		let me8  = me[ 8 ] , me9  = me[ 9 ] , me10 = me[ 10 ], me11 = me[ 11 ];
		let me12 = me[ 12 ], me13 = me[ 13 ], me14 = me[ 14 ], me15 = me[ 15 ];

		// see https://www8.cs.umu.se/kurser/5DV051/HT12/lab/plane_extraction.pdf
		planes[ 0 ].setComponents( me3 - me0, me7 - me4, me11 -  me8, me15 - me12 ).normalize(); // right
		planes[ 1 ].setComponents( me3 + me0, me7 + me4, me11 +  me8, me15 + me12 ).normalize(); // left
		planes[ 2 ].setComponents( me3 + me1, me7 + me5, me11 +  me9, me15 + me13 ).normalize(); // bottom
		planes[ 3 ].setComponents( me3 - me1, me7 - me5, me11 -  me9, me15 - me13 ).normalize(); // top
		planes[ 4 ].setComponents( me3 - me2, me7 - me6, me11 - me10, me15 - me14 ).normalize(); // far
		planes[ 5 ].setComponents( me3 + me2, me7 + me6, me11 + me10, me15 + me14 ).normalize(); // near

		// planes[ 5 ].setComponents( me2, me6, me10, me14 ).normalize();


		return this;

	}

	intersectsSphere(sphere) {

		const planes = this.planes;
		const center = sphere.center;
		const negRadius = - sphere.radius;

		for ( let i = 0; i < 6; i ++ ) {

			const distance = planes[ i ].distanceToPoint( center );

			if ( distance < negRadius ) {

				return false;

			}

		}

		return true;

	}

	intersectsBox( box ) {

		const planes = this.planes;

		for ( let i = 0; i < 6; i ++ ) 
		// for ( let i of [0, 1, 2, 3, 5]) 
		// for ( let i of [0]) 
		{

			const plane = planes[ i ];

			// corner at max distance

			_vector.x = plane.normal.x > 0 ? box.max.x : box.min.x;
			_vector.y = plane.normal.y > 0 ? box.max.y : box.min.y;
			_vector.z = plane.normal.z > 0 ? box.max.z : box.min.z;

			if ( plane.distanceToPoint( _vector ) < 0 ) {

				return false;

			}

		}

		return true;

	}

}