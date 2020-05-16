
// Taken and adapted from three.js: https://github.com/mrdoob/three.js
// LICENSE: MIT
// Copyright © 2010-2020 three.js authors

import {Quaternion} from "./Quaternion.js";

const {sqrt} = Math;

export class Vector3{

	constructor(x, y, z){
		this.x = x;
		this.y = y;
		this.z = z;
	}

	set(x, y, z){
		this.x = x;
		this.y = y;
		this.z = z;

		return this;
	}

	add(vec){
		this.x += vec.x;
		this.y += vec.y;
		this.z += vec.z;

		return this;
	}

	addScalar(scalar){
		this.x += scalar;
		this.y += scalar;
		this.z += scalar;

		return this;
	}

	sub(vec){
		this.x -= vec.x;
		this.y -= vec.y;
		this.z -= vec.z;

		return this;
	}

	subVectors(a, b){
		this.x = a.x - b.x;
		this.y = a.y - b.y;
		this.z = a.z - b.z;

		return this;
	}

	dot(v){
		return this.x * v.x + this.y * v.y + this.z * v.z;
	}

	length(){
		return sqrt( this.x * this.x + this.y * this.y + this.z * this.z );
	}

	lengthSq(){
		return this.x * this.x + this.y * this.y + this.z * this.z;
	}

	distanceTo(point){

		let dx = point.x - this.x;
		let dy = point.y - this.y;
		let dz = point.z - this.z;

		return sqrt(dx * dx + dy * dy + dz * dz);
	}

	normalize(){
		return this.divideScalar( this.length() || 1 );
	}

	crossVectors(a, b){
		let ax = a.x, ay = a.y, az = a.z;
		let bx = b.x, by = b.y, bz = b.z;

		this.x = ay * bz - az * by;
		this.y = az * bx - ax * bz;
		this.z = ax * by - ay * bx;

		return this;
	}


	multiplyScalar(scalar){
		this.x *= scalar;
		this.y *= scalar;
		this.z *= scalar;

		return this;
	}

	divideScalar(scalar){
		this.x /= scalar;
		this.y /= scalar;
		this.z /= scalar;

		return this;
	}

	applyQuaternion(q){
		// need to compute q * v * q⁻¹

		let {x, y, z} = this;
		let [qx, qy, qz, qw] = [q.x, q.y, q.z, q.w];

		// q * v
		let ix = qw * x + qy * z - qz * y;
		let iy = qw * y + qz * x - qx * z;
		let iz = qw * z + qx * y - qy * x;
		let iw = - qx * x - qy * y - qz * z;

		// result * q⁻¹
		this.x = ix * qw + iw * - qx + iy * - qz - iz * - qy;
		this.y = iy * qw + iw * - qy + iz * - qx - ix * - qz;
		this.z = iz * qw + iw * - qz + ix * - qy - iy * - qx;

		return this;
	}

	applyMatrix3( m ) {

		let {x, y, z} = this;
		let e = m.elements;

		this.x = e[ 0 ] * x + e[ 3 ] * y + e[ 6 ] * z;
		this.y = e[ 1 ] * x + e[ 4 ] * y + e[ 7 ] * z;
		this.z = e[ 2 ] * x + e[ 5 ] * y + e[ 8 ] * z;

		return this;
	}

	applyMatrix4( m ) {

		let {x, y, z} = this;
		let e = m.elements;

		let w = 1 / ( e[ 3 ] * x + e[ 7 ] * y + e[ 11 ] * z + e[ 15 ] );

		this.x = ( e[ 0 ] * x + e[ 4 ] * y + e[ 8 ] * z + e[ 12 ] ) * w;
		this.y = ( e[ 1 ] * x + e[ 5 ] * y + e[ 9 ] * z + e[ 13 ] ) * w;
		this.z = ( e[ 2 ] * x + e[ 6 ] * y + e[ 10 ] * z + e[ 14 ] ) * w;

		return this;
	}

	copy(vec){
		this.x = vec.x;
		this.y = vec.y;
		this.z = vec.z;

		return this;
	}

	clone(){
		return new Vector3(this.x, this.y, this.z);
	}

	toArray(){
		return [this.x, this.y, this.z];
	}


}