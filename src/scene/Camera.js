
import {Vector3} from "../math/Vector3.js";
import {Quaternion} from "../math/Quaternion.js";
import {toRadians} from "../math/mathtools.js";

export class Camera{

	constructor(){
		this.position = new Vector3(0, 0, 0);
		this.orientation = new Quaternion(0, 0, 0, 1);
		this.fov = toRadians(60);
		this.distance = 10;
		this.near = 0.1;
		this.far = 10_000;
	}

	lookAt(target){
		let up = vec3.fromValues(0, 0, 1);
	}

	getDirection(){
		let dir = new Vector3(0, 1, 0).applyQuaternion(this.orientation);

		return dir;
	}

	getTarget(){
		let dir = this.getDirection();
		let target = this.position.clone().add(dir.multiplyScalar(this.distance));

		return target;
	}

	getView(){
		let position = this.position;
		let target = this.getTarget();

		let up = [0, 0, 1];
		let view = mat4.create();
		mat4.lookAt(view, position.toArray(), target.toArray(), up);

		return view;
	}

	getProjection(aspect){

		let {near, far} = this;
		
		let proj = mat4.create();
		mat4.perspective(proj, 45, aspect, near, far);

		return proj;
	}

}