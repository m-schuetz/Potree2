
import {Vector3} from "../math/Vector3.js";
import {Matrix4} from "../math/Matrix4.js";
import {Quaternion} from "../math/Quaternion.js";
import {toRadians} from "../math/mathtools.js";

export class Camera{

	constructor(){
		this.position = new Vector3(0, 0, 0);
		this.orientation = new Quaternion(0, 0, 0, 1);
		this.fov = toRadians(60);
		this.distance = 10;
		this.near = 0.1;
		this.far = 1_000_000;
		this.width = 0;
		this.height = 0;
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

		let up = new Vector3(0, 0, 1);

		let view = new Matrix4();
		view.lookAt(position, target, up);

		return view;
	}

	getProjection(){

		let aspect = this.width / this.height;

		let {near, far, fov} = this;

		let top = near * Math.tan(0.5 * fov);
		let height = 2 * top;
		let width = aspect * height;
		let left = - 0.5 * width;

		let proj = new Matrix4();
		proj.makePerspective( 
			left, left + width, 
			top, top - height, 
			near, far);

		// {
		// 	let proj = mat4.create();
		// 	mat4.perspective(proj, 45, aspect, near, far);

		// 	let a = 10;
		// }

		return proj;
	}

}