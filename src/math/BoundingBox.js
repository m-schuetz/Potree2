
import {Vector3} from "./Vector3.js";

export class BoundingBox{

	constructor(min, max){
		this.min = min;
		this.max = max;
	}

	size(){
		return new Vector3(
			this.max.x - this.min.x,
			this.max.y - this.min.y,
			this.max.z - this.min.z,
		);
	}

	center(){
		return new Vector3(
			(this.min.x + this.max.x) / 2,
			(this.min.y + this.max.y) / 2,
			(this.min.z + this.max.z) / 2,
		);
	}

}