
import {Vector3} from "../math/Vector3.js";
import {Quaternion} from "../math/Quaternion.js";
import {toRadians} from "../math/mathtools.js";

export class Camera{

	constructor(){
		this.position = new Vector3(0, 0, 0);
		this.orientation = new Quaternion(0, 0, 0, 1);
		this.fov = toRadians(60);
		this.distance = 10;
		this.near = 0;
		this.far = 10_000;
	}

	lookAt(target){
		let up = vec3.fromValues(0, 0, 1);
	}

}