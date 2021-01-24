
import {Vector3, Matrix4} from "../math/math.js";

export class OrbitControls{

	constructor(element){

		this.element = element;
		this.radius = 5;
		this.yaw = 0;
		this.pitch = 0;
		this.pivot = new Vector3();
		this.world = new Matrix4();

		// this.world = mat4.create();

		element.addEventListener('contextmenu', e => {
			e.preventDefault();

			return false;
		});

		element.addEventListener('mousemove', e => {

			let dragLeft = e.buttons === 1;
			let dragRight = e.buttons === 2;

			if(dragLeft){
				let diffX = e.movementX;
				let diffY = e.movementY;

				let ux = diffX / this.element.width;
				let uy = diffY / this.element.height;

				this.yaw += 6 * ux;
				this.pitch += 6 * uy;
			}else if(dragRight){
				let diffX = e.movementX;
				let diffY = e.movementY;

				let ux = diffX / this.element.width;
				let uy = diffY / this.element.height;

				this.translate_local(
					-ux * this.radius, 
					0, 
					uy * this.radius);

			}
		});

		element.addEventListener('wheel', e => {
			let diff = Math.sign(e.deltaY);

			if(diff > 0){
				this.radius *= 1.05;
			}else if(diff < 0){
				this.radius /= 1.05;
			}

		});
	}

	set(args){
		this.yaw = args.yaw ?? this.yaw;
		this.pitch = args.pitch ?? this.pitch;
		this.radius = args.radius ?? this.radius;

		if(args.pivot){
			this.pivot.set(...args.pivot);
		}
	}

	getPosition(){
		return new Vector3().applyMatrix4(this.world);
	}

	translate_local(x, y, z){
		let _pos = new Vector3(0, 0, 0);
		let _right = new Vector3(1, 0, 0);
		let _forward = new Vector3(0, 1, 0);
		let _up = new Vector3(0, 0, 1);
		
		_pos.applyMatrix4(this.world);
		_right.applyMatrix4(this.world);
		_forward.applyMatrix4(this.world);
		_up.applyMatrix4(this.world);

		_right.sub(_pos).normalize();
		_forward.sub(_pos).normalize();
		_up.sub(_pos).normalize();

		_right.multiplyScalar(x);
		_forward.multiplyScalar(z);
		_up.multiplyScalar(-y);

		this.pivot.add(_right);
		this.pivot.add(_forward);
		this.pivot.add(_up);
	}

	update(delta){

		let flip = new Matrix4().set(
			1, 0, 0, 0,
			0, 0, 1, 0,
			0, -1, 0, 0,
			0, 0, 0, 1,
		);

		this.world.makeIdentity();
		this.world.translate(0, 0, this.radius);
		this.world.multiplyMatrices(flip, this.world);
		this.world.rotate(Math.PI / 2 - this.pitch, new Vector3(1, 0, 0));
		this.world.rotate(-this.yaw, new Vector3(0, 1, 0));

		this.world.translate(this.pivot.x, this.pivot.z, -this.pivot.y);

		{
			let flip = new Matrix4().set(
				1, 0, 0, 0,
				0, 0, -1, 0,
				0, 1, 0, 0,
				0, 0, 0, 1,
			);

			this.world.multiplyMatrices(flip, this.world);
		}

	}



};