
// Taken and adapted from three.js: https://github.com/mrdoob/three.js
// LICENSE: MIT
// Copyright © 2010-2020 three.js authors

// Also taken and adapted from gl-matrix.js

import {Vector3} from "./Vector3.js";

export class Matrix3{

	constructor(){
		this.elements = [
			1, 0, 0,
			0, 1, 0,
			0, 0, 1,
		];
	}

	set(n11, n12, n13, n21, n22, n23, n31, n32, n33){
		let te = this.elements;

		te[ 0 ] = n11; te[ 1 ] = n21; te[ 2 ] = n31;
		te[ 3 ] = n12; te[ 4 ] = n22; te[ 5 ] = n32;
		te[ 6 ] = n13; te[ 7 ] = n23; te[ 8 ] = n33;

		return this;
	}

	getNormalMatrix( matrix4 ) {
		return this.setFromMatrix4( matrix4 ).getInverse( this ).transpose();
	}

	setFromMatrix4( m ) {
		let me = m.elements;

		this.set(
			me[ 0 ], me[ 4 ], me[ 8 ],
			me[ 1 ], me[ 5 ], me[ 9 ],
			me[ 2 ], me[ 6 ], me[ 10 ]
		);

		return this;
	}

	transpose(){

		let tmp;
		let m = this.elements;

		tmp = m[ 1 ]; m[ 1 ] = m[ 3 ]; m[ 3 ] = tmp;
		tmp = m[ 2 ]; m[ 2 ] = m[ 6 ]; m[ 6 ] = tmp;
		tmp = m[ 5 ]; m[ 5 ] = m[ 7 ]; m[ 7 ] = tmp;

		return this;

	}

	getInverse( matrix ) {

		let me = matrix.elements;
		let te = this.elements;

		let n11 = me[ 0 ], n21 = me[ 1 ], n31 = me[ 2 ];
		let n12 = me[ 3 ], n22 = me[ 4 ], n32 = me[ 5 ];
		let n13 = me[ 6 ], n23 = me[ 7 ], n33 = me[ 8 ];

		let t11 = n33 * n22 - n32 * n23;
		let t12 = n32 * n13 - n33 * n12;
		let t13 = n23 * n12 - n22 * n13;

		let det = n11 * t11 + n21 * t12 + n31 * t13;

		if ( det === 0 ){
			return this.set( 0, 0, 0, 0, 0, 0, 0, 0, 0 );
		}

		let detInv = 1 / det;

		te[ 0 ] = t11 * detInv;
		te[ 1 ] = ( n31 * n23 - n33 * n21 ) * detInv;
		te[ 2 ] = ( n32 * n21 - n31 * n22 ) * detInv;

		te[ 3 ] = t12 * detInv;
		te[ 4 ] = ( n33 * n11 - n31 * n13 ) * detInv;
		te[ 5 ] = ( n31 * n12 - n32 * n11 ) * detInv;

		te[ 6 ] = t13 * detInv;
		te[ 7 ] = ( n21 * n13 - n23 * n11 ) * detInv;
		te[ 8 ] = ( n22 * n11 - n21 * n12 ) * detInv;

		return this;

	}
}