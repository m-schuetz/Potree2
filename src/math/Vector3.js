
import {Quaternion} from "./Quaternion.js";

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

	multiplyScalar(scalar){
		this.x *= scalar;
		this.y *= scalar;
		this.z *= scalar;

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