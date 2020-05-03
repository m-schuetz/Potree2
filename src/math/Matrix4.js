
export class Matrix4{

	constructor(){
		this.elements = [
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		];
	}

	setFromQuaternion(quaternion){
		let e = this.elements;

		let {x, y, z, w} = quaternion;
		let xx = x * (x + x);
		let xy = x * (y + y);
		let xz = x * (z + z);
		let yy = y * (y + y);
		let yz = y * (z + z);
		let zz = z * (z + z);
		let wx = w * (x + x);
		let wy = w * (y + y);
		let wz = w * (z + z);

		e[0] = 1 - (yy + zz);
		e[1] = xy + wz;
		e[2] = xz - wy;
		e[3] = 0;

		e[4] = xy - wz;
		e[5] = 1 - (xx + zz);
		e[6] = yz + wx;
		e[7] = 0;

		e[8] = xz + wy;
		e[9] = yz - wx;
		e[10] = 1 - (xx + yy);
		e[11] = 0;

		e[12] = 0;
		e[13] = 0;
		e[14] = 0;
		e[15] = 1;
	}

}