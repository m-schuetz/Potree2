
// Taken and adapted from three.js: https://github.com/mrdoob/three.js
// LICENSE: MIT
// Copyright © 2010-2020 three.js authors

export class Quaternion{

	constructor(x, y, z, w){
		this.x = x;
		this.y = y;
		this.z = z;
		this.w = w;
	}

	multiplyQuaternions(a, b){
		// from http://www.euclideanspace.com/maths/algebra/realNormedAlgebra/quaternions/code/index.htm#mul
		this.x =  a.x * b.w + a.y * b.z - a.z * b.y + a.w * b.x;
		this.y = -a.x * b.z + a.y * b.w + a.z * b.x + a.w * b.y;
		this.z =  a.x * b.y - a.y * b.x + a.z * b.w + a.w * b.z;
		this.w = -a.x * b.x - a.y * b.y - a.z * b.z + a.w * b.w;

		return this;
	}

	setFromEuler(x, y, z){

		let {sin, cos} = Math;

		let s1 = sin(x / 2);
		let s2 = sin(y / 2);
		let s3 = sin(z / 2);

		let c1 = cos(x / 2);
		let c2 = cos(y / 2);
		let c3 = cos(z / 2);

		this.x = s1 * c2 * c3 + c1 * s2 * s3;
		this.y = c1 * s2 * c3 - s1 * c2 * s3;
		this.z = c1 * c2 * s3 + s1 * s2 * c3;
		this.w = c1 * c2 * c3 - s1 * s2 * s3;

		return this;
	}

}