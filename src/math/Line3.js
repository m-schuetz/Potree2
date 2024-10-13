// 
// Adapted from three.js
// license: MIT (https://github.com/mrdoob/three.js/blob/dev/LICENSE)
// url: https://github.com/mrdoob/three.js/blob/dev/src/math/Line3.js
//

import {Vector3} from "./Vector3.js";

const _startP = new Vector3();
const _startEnd = new Vector3();

export class Line3{

	constructor(start, end){
		this.start = start ?? new Vector3();
		this.end = end ?? new Vector3();
	}

	closestPointToPointParameter( point, clampToLine ) {

		_startP.subVectors( point, this.start );
		_startEnd.subVectors( this.end, this.start );

		const startEnd2 = _startEnd.dot( _startEnd );
		const startEnd_startP = _startEnd.dot( _startP );

		let t = startEnd_startP / startEnd2;

		if ( clampToLine ) {

			t = MathUtils.clamp( t, 0, 1 );

		}

		return t;

	}

	closestPointToPoint(point, clampToLine) {
		const t = this.closestPointToPointParameter( point, clampToLine );

		return this.delta( target ).multiplyScalar( t ).add( this.start );
	}

	delta( target ) {
		return target.subVectors( this.end, this.start );
	}

	at(t, target){
		return this.delta( target ).multiplyScalar( t ).add( this.start );
	}

}

