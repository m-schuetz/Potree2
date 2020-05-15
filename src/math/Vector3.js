
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