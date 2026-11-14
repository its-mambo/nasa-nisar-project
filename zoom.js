"use strict";

class RasterViewport {
  constructor(canvas, redraw) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.redraw = redraw;
    this.width = this.height = this.w = this.h = 0;
    this.x = this.y = 0;
    this.scale = this.fitScale = 1;
    this.pending = false;
    this.drag = null;
    new ResizeObserver(() => this.invalidate()).observe(canvas);

    canvas.addEventListener("wheel", e => {
      const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? this.h : 1;
      if (e.ctrlKey) {
        e.preventDefault();
        if (!this.width || !this.w) return;
        const [x, y] = this.local(e), [col, row] = this.pixel(x, y);
        this.scale = Math.max(this.fitScale, Math.min(8,
          this.scale * Math.exp(-Math.max(-240, Math.min(240, e.deltaY * unit)) / 300)));
        this.x = x - col * this.scale;
        this.y = y - row * this.scale;
        this.clamp();
        this.invalidate();
      } else {
        const dx = e.shiftKey && !e.deltaX ? e.deltaY : e.deltaX;
        const dy = e.shiftKey && !e.deltaX ? 0 : e.deltaY;
        if ((dx && this.width * this.scale > this.w) || (dy && this.height * this.scale > this.h)) {
          e.preventDefault();
          this.pan(-dx * unit, -dy * unit);
        }
      }
    }, {passive:false});
  }

  setRaster(width, height) {
    this.width = width;
    this.height = height;
    this.size();
    // size() returns early when the canvas is unchanged, so refit for the new raster here.
    if (width && height && this.w && this.h) this.fitScale = Math.min(1, this.w/width, this.h/height);
    this.scale = this.fitScale;
    this.x = this.y = Infinity;
    this.clamp();
    this.invalidate();
  }

  size() {
    const {width:w, height:h} = this.canvas.getBoundingClientRect();
    if (!w || !h) return false;
    const dpr = window.devicePixelRatio || 1;
    const cw = Math.max(1, Math.round(w*dpr)), ch = Math.max(1, Math.round(h*dpr));
    if (this.canvas.width !== cw) this.canvas.width = cw;
    if (this.canvas.height !== ch) this.canvas.height = ch;
    if (this.w === w && this.h === h) return true;
    const fitted = this.scale <= this.fitScale + 1e-9;
    const [col, row] = this.pixel(this.w/2, this.h/2);
    this.w = w;
    this.h = h;
    if (this.width && this.height) {
      this.fitScale = Math.min(1, w/this.width, h/this.height);
      this.scale = fitted ? this.fitScale : Math.max(this.fitScale, this.scale);
      this.x = w/2 - col*this.scale;
      this.y = h/2 - row*this.scale;
      this.clamp();
    }
    return true;
  }

  clamp() {
    const minX = this.w - this.width*this.scale, minY = this.h - this.height*this.scale;
    this.x = minX >= 0 ? minX/2 : Math.max(minX, Math.min(0, this.x));
    this.y = minY >= 0 ? minY/2 : Math.max(minY, Math.min(0, this.y));
  }

  local(e) {
    const r = this.canvas.getBoundingClientRect();
    return [e.clientX-r.left, e.clientY-r.top];
  }

  pixel(x, y) { return [(x-this.x)/this.scale, (y-this.y)/this.scale]; }

  visible() {
    const [x, y] = this.pixel(0, 0), [X, Y] = this.pixel(this.w, this.h);
    return [Math.max(0,x), Math.max(0,y), Math.min(this.width,X), Math.min(this.height,Y)];
  }

  zoomBy(factor, x = this.w/2, y = this.h/2) {
    if (!this.width || !this.w) return;
    const [col, row] = this.pixel(x, y);
    this.scale = Math.max(this.fitScale, Math.min(8, this.scale * factor));
    this.x = x - col * this.scale;
    this.y = y - row * this.scale;
    this.clamp();
    this.invalidate();
  }

  fit() {
    this.scale = this.fitScale;
    this.clamp();
    this.invalidate();
  }

  pan(dx, dy) {
    this.x += dx;
    this.y += dy;
    this.clamp();
    this.invalidate();
  }

  enableDrag(button) {
    const canvas = this.canvas;
    canvas.addEventListener("pointerdown", e => {
      if (!e.isPrimary || e.button !== button || !this.width) return;
      e.preventDefault();
      this.drag = [e.pointerId, e.clientX, e.clientY];
      canvas.setPointerCapture(e.pointerId);
      canvas.classList.add("panning");
    });
    canvas.addEventListener("pointermove", e => {
      if (this.drag?.[0] !== e.pointerId) return;
      this.pan(e.clientX - this.drag[1], e.clientY - this.drag[2]);
      this.drag = [e.pointerId, e.clientX, e.clientY];
    });
    canvas.addEventListener("lostpointercapture", e => {
      if (this.drag?.[0] === e.pointerId) {
        this.drag = null;
        canvas.classList.remove("panning");
      }
    });
  }

  invalidate() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => {
      this.pending = false;
      if (!this.size()) return;
      const {ctx, canvas, scale} = this;
      ctx.resetTransform();
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!this.width) return;
      ctx.scale(canvas.width/this.w, canvas.height/this.h);
      ctx.translate(this.x, this.y);
      ctx.scale(scale, scale);
      ctx.imageSmoothingEnabled = scale < 1;
      this.redraw(ctx);
    });
  }
}
