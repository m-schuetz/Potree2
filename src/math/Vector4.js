// 
// Adapted from three.js
// license: MIT (https://github.com/mrdoob/three.js/blob/dev/LICENSE)
// url: https://github.com/mrdoob/three.js/blob/dev/src/math/Vector3.js
//

export class Vector4{

	constructor(x, y, z, w){
		this.x = x ?? 0;
		this.y = y ?? 0;
		this.z = z ?? 0;
		this.w = w ?? 0;
	}

	set(x, y, z, w){
		this.x = x;
		this.y = y;
		this.z = z;
		this.w = w;

		return this;
	}

	copy(b){
		this.x = b.x;
		this.y = b.y;
		this.z = b.z;
		this.w = b.w;

		return this;
	}

	multiplyScalar(s){
		this.x = this.x * s;
		this.y = this.y * s;
		this.z = this.z * s;
		this.w = this.w * s;

		return this;
	}

	divideScalar(s){
		this.x = this.x / s;
		this.y = this.y / s;
		this.z = this.z / s;
		this.w = this.w / s;

		return this;
	}

	add(b){
		this.x = this.x + b.x;
		this.y = this.y + b.y;
		this.z = this.z + b.z;
		this.w = this.w + b.w;

		return this;
	}

	addScalar(s){
		this.x = this.x + s;
		this.y = this.y + s;
		this.z = this.z + s;
		this.w = this.w + s;

		return this;
	}

	sub(b){
		this.x = this.x - b.x;
		this.y = this.y - b.y;
		this.z = this.z - b.z;
		this.w = this.w - b.w;

		return this;
	}

	subScalar(s){
		this.x = this.x - s;
		this.y = this.y - s;
		this.z = this.z - s;
		this.w = this.w - s;

		return this;
	}

	subVectors( a, b ) {

		this.x = a.x - b.x;
		this.y = a.y - b.y;
		this.z = a.z - b.z;
		this.w = a.w - b.w;

		return this;
	}

	cross(v) {
		return this.crossVectors( this, v );
	}

	// crossVectors(a, b) {

	// 	const ax = a.x, ay = a.y, az = a.z;
	// 	const bx = b.x, by = b.y, bz = b.z;

	// 	this.x = ay * bz - az * by;
	// 	this.y = az * bx - ax * bz;
	// 	this.z = ax * by - ay * bx;

	// 	return this;
	// }

	dot( v ) {
		return this.x * v.x + this.y * v.y + this.z * v.z + this.w * v.w;
	}

	distanceTo( v ) {
		return Math.sqrt( this.distanceToSquared( v ) );
	}

	distanceToSquared( v ) {
		const dx = this.x - v.x;
		const dy = this.y - v.y;
		const dz = this.z - v.z;
		const dw = this.w - v.w;

		return dx * dx + dy * dy + dz * dz + dw * dw;
	}

	clone(){
		return new Vector4(this.x, this.y, this.z, this.w);
	}

	applyMatrix4(m){
		const {x, y, z, w} = this;
		const e = m.elements;

		this.x = e[0] * x + e[4] * y + e[ 8] * z + e[12] * w;
		this.y = e[1] * x + e[5] * y + e[ 9] * z + e[13] * w;
		this.z = e[2] * x + e[6] * y + e[10] * z + e[14] * w;
		this.w = e[3] * x + e[7] * y + e[11] * z + e[15] * w;

		return this;
	}

	length() {
		return Math.sqrt( this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w );
	}

	lengthSq() {
		return this.x * this.x + this.y * this.y + this.z * this.z + this.w * this.w;
	}

	normalize(){
		let l = this.length();

		this.x = this.x / l;
		this.y = this.y / l;
		this.z = this.z / l;
		this.w = this.w / l;

		return this;
	}

	toString(precision){
		if(precision != null){
			return `${this.x.toFixed(precision)}, ${this.y.toFixed(precision)}, ${this.z.toFixed(precision)}, ${this.w.toFixed(precision)}`;
		}else{
			return `${this.x}, ${this.y}, ${this.z}, ${this.w}`;
		}
	}

	toArray(){
		return [this.x, this.y, this.z, this.w];
	}

	isFinite(){
		let {x, y, z, w} = this;
		
		return Number.isFinite(x) && Number.isFinite(y) && Number.isFinite(z) && Number.isFinite(w);
	}

};