
// Taken and adapted from three.js: https://github.com/mrdoob/three.js
// LICENSE: MIT
// Copyright © 2010-2020 three.js authors

import { Vector3 } from './Vector3.js';
import { Plane } from './Plane.js';

const _vector = new Vector3();

export class Frustum{
	constructor(){
		this.planes = [
			new Plane(), new Plane(), new Plane(), 
			new Plane(), new Plane(), new Plane(),
		];
	}

	setFromProjectionMatrix(m){

		let planes = this.planes;

		let [
			me0, me1, me2, me3,
			me4, me5, me6, me7,
			me8, me9, me10, me11,
			me12, me13, me14, me15
		] = m.elements;

		planes[0].setComponents( me3 - me0, me7 - me4, me11 - me8, me15 - me12 ).normalize();
		planes[1].setComponents( me3 + me0, me7 + me4, me11 + me8, me15 + me12 ).normalize();
		planes[2].setComponents( me3 + me1, me7 + me5, me11 + me9, me15 + me13 ).normalize();
		planes[3].setComponents( me3 - me1, me7 - me5, me11 - me9, me15 - me13 ).normalize();
		planes[4].setComponents( me3 - me2, me7 - me6, me11 - me10, me15 - me14 ).normalize();
		planes[5].setComponents( me3 + me2, me7 + me6, me11 + me10, me15 + me14 ).normalize();

		return this;
	}

	intersectsBox(box){
		let planes = this.planes;

		for(let i = 0; i < 6; i ++ ) {

			let plane = planes[ i ];

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

	applyMatrix4(matrix){
		for(let plane of this.planes){
			plane.applyMatrix4(matrix);
		}
	}


}