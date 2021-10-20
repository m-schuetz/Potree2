
import {Vector3, Matrix4, Box3} from "potree";
import {Potree, EventDispatcher} from "potree";
import * as TWEEN from "tween";

export class OrbitControls{

	constructor(element){

		this.element = element;
		this.radius = 5;
		this.yaw = 0;
		this.pitch = 0;
		this.pivot = new Vector3();
		this.world = new Matrix4();
		this.dispatcher = new EventDispatcher();

		this.dispatcher.add('mousemove', e => {

			let dragLeft = e.event.buttons === 1;
			let dragRight = e.event.buttons === 2;

			if(dragLeft){
				let diffX = e.event.movementX;
				let diffY = e.event.movementY;

				let ux = diffX / this.element.width;
				let uy = diffY / this.element.height;

				this.yaw += 6 * ux;
				this.pitch += 6 * uy;
			}else if(dragRight){
				let diffX = e.event.movementX;
				let diffY = e.event.movementY;

				let ux = diffX / this.element.width;
				let uy = diffY / this.element.height;

				this.translate_local(
					-ux * this.radius, 
					0, 
					uy * this.radius);

			}
		});

		this.dispatcher.add('mousewheel', e => {
			let diff = -Math.sign(e.delta);

			if(diff > 0){
				this.radius *= 1.1;
			}else if(diff < 0){
				this.radius /= 1.1;
			}

		});

		this.dispatcher.add("dblclick", e => {

			let {x, y} = e.mouse;

			Potree.pick(x, y, (result) => {
				let newRadius = result.depth * 0.25;
				let newPivot = result.position;

				// this.set({radius: newRadius, pivot: newPivot});

				let value = {x: 0};
				let animationDuration = 400;
				let easing = TWEEN.Easing.Quartic.Out;
				let tween = new TWEEN.Tween(value).to({x: 1}, animationDuration);
				tween.easing(easing);
				// this.tweens.push(tween);

				// let startPos = this.getPosition();
				// let targetPos = cameraTargetPosition.clone();
				let startRadius = this.radius;
				let targetRadius = newRadius;
				let startPivot = this.pivot.clone();
				let targetPivot = newPivot;

				tween.onUpdate(() => {
					let t = value.x;

					let pivot = new Vector3(
						(1 - t) * startPivot.x + t * targetPivot.x,
						(1 - t) * startPivot.y + t * targetPivot.y,
						(1 - t) * startPivot.z + t * targetPivot.z,
					);

					let radius = (1 - t) * startRadius + t * targetRadius;

					// this.viewer.setMoveSpeed(this.scene.view.radius);
					this.set({radius: radius, pivot: pivot});
				});

				tween.onComplete(() => {
					// this.tweens = this.tweens.filter(e => e !== tween);
				});

				tween.start();

			});

		});
	}

	set({yaw, pitch, radius, pivot, position}){
		this.yaw = yaw ?? this.yaw;
		this.pitch = pitch ?? this.pitch;
		this.radius = radius ?? this.radius;

		if(pivot){
			if(typeof pivot.x !== "undefined"){
				this.pivot.copy(pivot);
			}else{
				this.pivot.set(...pivot);
			}
		}
		
		if(position !== undefined && pivot !== undefined){
			let diff = new Vector3(
				pivot[0] - position[0],
				pivot[1] - position[1],
				pivot[2] - position[2],
			);

			let radius = diff.length();
			let yaw = Math.PI / 2 - Math.atan2(diff.y, diff.x);
			let groundRadius = Math.sqrt(diff.x ** 2 + diff.y ** 2);
			let pitch = -Math.atan2(diff.z, groundRadius);

			this.yaw = yaw;
			this.pitch = pitch;
			this.radius = radius;

		} 
	}

	zoomTo(node, args){

		let box = new Box3();
		let tmp = new Box3();
		node.traverse((node) => {

			let childBox = node.boundingBox;

			if(!childBox.isFinite()){
				return;
			}

			tmp.copy(childBox);
			tmp.applyMatrix4(node.world);
			
			box.expandByBox(tmp);
		});

		let pivot = box.center();
		let multiplier = args.zoom ?? 1.0;
		let radius = box.size().length() * 0.8 * multiplier;

		this.set({pivot, radius});

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

	toExpression(){

		let pivot = this.pivot;

		let str = `;
		controls.set(
			yaw: ${this.yaw},
			pitch: ${this.pitch},
			radius: ${this.radius},
			pivot: new Vector3(${pivot.x}, ${pivot.y}, ${pivot.z}),
		);
		`;

		return str;
	}



};