
export class Matrix4{

	constructor(){
		this.elements = [
			1, 0, 0, 0,
			0, 1, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1,
		];
	}

	set(n11, n12, n13, n14, n21, n22, n23, n24, n31, n32, n33, n34, n41, n42, n43, n44 ){
		let te = this.elements;

		te[ 0 ] = n11; te[ 4 ] = n12; te[ 8 ] = n13; te[ 12 ] = n14;
		te[ 1 ] = n21; te[ 5 ] = n22; te[ 9 ] = n23; te[ 13 ] = n24;
		te[ 2 ] = n31; te[ 6 ] = n32; te[ 10 ] = n33; te[ 14 ] = n34;
		te[ 3 ] = n41; te[ 7 ] = n42; te[ 11 ] = n43; te[ 15 ] = n44;

		return this;
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

	makeRotationZ(theta){

		let c = Math.cos(theta);
		let s = Math.sin(theta);

		this.set(

			c, -s, 0, 0,
			s, c, 0, 0,
			0, 0, 1, 0,
			0, 0, 0, 1

		);

		return this;
	}

}