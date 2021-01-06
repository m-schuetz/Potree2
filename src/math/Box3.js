// 
// Adapted from three.js
// license: MIT (https://github.com/mrdoob/three.js/blob/dev/LICENSE)
// url: https://github.com/mrdoob/three.js/blob/dev/src/math/Box3.js
//

import {Vector3} from "./Vector3.js";

export class Box3{

	constructor(min, max){
		this.min = min ?? new Vector3(+Infinity, +Infinity, +Infinity);
		this.max = max ?? new Vector3(-Infinity, -Infinity, -Infinity);
	}

	clone(){
		return new Box3(
			this.min.clone(), 
			this.max.clone()
		);
	}

	size(){
		return this.max.clone().sub(this.min);
	}

	expandByPoint(point){
		this.min.x = Math.min(this.min.x, point.x);
		this.min.y = Math.min(this.min.y, point.y);
		this.min.z = Math.min(this.min.z, point.z);

		this.max.x = Math.max(this.max.x, point.x);
		this.max.y = Math.max(this.max.y, point.y);
		this.max.z = Math.max(this.max.z, point.z);
	}

};