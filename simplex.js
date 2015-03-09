function SimplexNoise(stdlib, foreign, heap) {
  "use asm";

  var fround = stdlib.Math.fround;
  var sqrt = stdlib.Math.sqrt;
  var pow = stdlib.Math.pow;
  var imul = stdlib.Math.imul;

  var HEAPF32 = new stdlib.Float32Array(heap);
  var HEAPI32 = new stdlib.Int32Array(heap);
  var HEAPU8 = new stdlib.Uint8Array(heap);

  /* Skew factor for dim dimensions.
   * @return {float} Fn
   */
  function F(dim) {
    dim = dim|0;
    var d = 0.;
    d = +(~~dim);
    return fround((+sqrt(1. + d) - 1.) / +d);
  }

  /* Unskew factor for dim dimensions.
   * @return {float} Gn
   */
  function G(dim) {
    dim = dim|0;

    var d0 = 0., d1 = 0.;
    d0 = +(~~dim);
    d1 = d0+1.;

    return fround((d1 - +sqrt(d1)) / +(d0 * d1));
  }

  function gradinit(dim, gp) {
    dim = dim|0;
    gp  = gp|0;

    var i = 0;

    for (i=(dim-1)|0; ~~i >= 0; i=(i-1)|0) {
      HEAPF32[(gp + (i<<2)) >> 2] = fround(0.);
    }
  }

  function gradstep(dim, mag, pp, gp) {
    dim = dim|0;
    mag = fround(mag);
    pp  = pp|0;
    gp  = gp|0;

    var i = 0, carry = 1, nzero = 0;
    var value = fround(0.);

    for (i=(dim-1)|0; ~~i >= 0; i=(i-1)|0) {
      value = fround(HEAPF32[(pp + (i<<2)) >> 2]);
      if (carry) {
        if (value == fround(0.)) {
          value = mag;
          carry = 0;
        } else if (value > fround(0.)) {
          value = fround(-mag);
          carry = 0;
        } else {
          value = fround(0.);
          nzero = (nzero + 1)|0;
        }
      } else if (value == fround(0.)) {
        nzero = (nzero + 1)|0;
      }
      HEAPF32[(gp + (i<<2)) >> 2] = value;
    }
    return nzero|0;
  }

  /* Build the gradient for the given number of dimensions.
   */
  function gradgen(dim, gp) {
    dim = dim|0;
    gp  = gp|0;

    var pp = 0, num = 0, nzero = 0;
    var mag = fround(0.);

    gradinit(dim, gp);
    pp = gp;

    mag =  fround(1.); // fround(fround(1.) / fround(sqrt(fround(~~(dim-1)))));

    while (1) {
      nzero = gradstep(dim, mag, pp, gp)|0;
      pp = gp;
      if (!((nzero-1)|0)) {
        num = (num+1)|0;
        gp = (dim << 2) + gp|0;
      } else if (!((nzero-dim)|0)) {
        break;
      }
    }
    return num|0;
  }

  function grad(dim, vp) {
    dim = dim|0;
    vp = vp|0;
    return +1;
  }

  /* Scalar (aka. dot-) product of two vectors.
   * @param {int} dim dimensions of the two vectors
   * @param {int} v0p pointer to float elements of v0
   * @param {int} v1p pointer to float elements of v1
   * @returns SUM 0 <= i < dim ( v0[i] * v1[i] )
   */
  function dot(dim, v0p, v1p) {
    dim = dim|0;
    v0p = v0p|0;
    v1p = v1p|0;

    var sum = fround(0.);
    var i = 0;

    for (i=dim; i; i=(i-1)|0) {
      sum = fround(fround(HEAPF32[v0p >> 2] * HEAPF32[v1p >> 2]) + sum);
      v0p = (v0p + 4)|0;
      v1p = (v1p + 4)|0;
    }
    return fround(sum);
  }

  function noise(dim, gp, gs, pp, ps, vp, rp, ip, dp) {
    dim = dim|0;
    gp  = gp|0; // gradient map
    gs  = gs|0; // gradient map size
    pp  = pp|0; // permutation map
    ps  = ps|0; // permutation map size
    vp  = vp|0; // position vector
    rp  = rp|0; // ranking vector (16 bytes)
    ip  = ip|0; // index vector (16*dim bytes)
    dp  = dp|0; // distance vector (16*(dim+1) bytes)

    var i = 0, j = 0, l = 0, p = 0, q = 0;
    var s = fround(0.);
    var t = fround(0.);
    var n = fround(0.);
    var d = fround(0.);
    var v = fround(0.);
    var Fn = fround(0.);
    var Gn = fround(0.);
    var nGn = fround(0.);

    Fn = fround(F(dim));
    Gn = fround(G(dim));

    // compute skewing factor s
    for (i=0; (dim-i)|0; i=(i+1)|0) {
      s = fround(HEAPF32[(vp + (i<<2)) >> 2] + s);
    }
    s = fround(Fn * s);

    // compute unskewing factor t
    for (i=0; (dim-i)|0; i=(i+1)|0) {
      // apply skewing transform
      j = ~~fround(fround(HEAPF32[(vp + (i<<2)) >> 2]) + s);
      // store coordinates in int heap
      HEAPI32[(ip + (i<<2)) >> 2] = j;
      // also, initialize ranking vector
      HEAPI32[(rp + (i<<2)) >> 2] = 0;

      t = fround(fround(~~j) + t);
    }
    t = fround(Gn * t);

    for (i=0; (dim-i)|0; i=(i+1)|0) {
      j = HEAPI32[(ip + (i<<2)) >> 2]|0;
      // compute distance from cell origin
      HEAPF32[(dp + (i<<2)) >> 2] = ( HEAPF32[(vp + (i<<2)) >> 2]
                                    - fround(fround(~~j) - t));
      // store cell coordinates
      HEAPI32[(ip + (i<<2)) >> 2] = j;

      // determine magnitude ordering of coordinates
      // compare starting from the first vector element
      // each element with the preceding ones
      for (j=0; (i-j)|0; j=(j+1)|0) {
        if (fround(HEAPF32[(dp + (i<<2)) >> 2]) > fround(HEAPF32[(dp + (j<<2)) >> 2])) {
          HEAPI32[(rp + (i<<2)) >> 2] = (HEAPI32[(rp + (i<<2)) >> 2]|0) + 1;
        } else {
          HEAPI32[(rp + (j<<2)) >> 2] = (HEAPI32[(rp + (j<<2)) >> 2]|0) + 1;
        }
      }
    }

    p = 0;
    for (j=0; (dim-j+1)|0; j=(j+1)|0) {
      n = fround(0.6);
      l = 0;
      for (i=0; (dim-i)|0; i=(i+1)|0) {
        q = HEAPI32[(ip + (i<<2)) >> 2]|0;
        d = fround(HEAPF32[(dp + (i<<2)) >> 2]);

        if (j) {
          // compute relative offsets of the other corners
          // based on ranking
          if ((HEAPI32[(rp + (i<<2)) >> 2]|0) >= ((dim-j)|0)) {
            q = (q + 1)|0;
            d = fround(fround(d - fround(1.)) + nGn);
          } else {
            q = q;
            d = fround(d + nGn);
          }

          HEAPF32[(dp + p + (i<<2)) >> 2] = d;
        }

        l = HEAPU8[(pp + l + q)|0]|0;
        // substract squared distances
        n = fround(n - fround(d*d));
      }
      if (n<fround(0.)) {
        n = fround(0.);
      } else {
        l = imul(~~l % ~~gs, dim) << 2; // convert gradent (mod gradient size) index to byte offset
        n = fround(n * n); // square
        n = fround(n * n); // again
        n = fround(n * fround(dot(dim, (gp + l)|0, (dp + p)|0)));
      }
      nGn = fround(nGn + Gn);
      p = (p + (dim<<2))|0;
      v = fround(v + n);
    }

    return fround(v);
  }

  return {
    F: F,
    G: G,
    dot: dot,
    grad: grad,
    gradgen: gradgen,
    noise: noise
  };
}

SimplexNoise.perm = new Uint8Array((function () {
  var p = [
    151,160,137,91,90,15,131,13,201,95,96,53,194,233,7,225,140,36,103,30,69,
    142,8,99,37,240,21,10,23,190,6,148,247,120,234,75,0,26,197,62,94,252,219,
    203,117,35,11,32,57,177,33,88,237,149,56,87,174,20,125,136,171,168,68,175,
    74,165,71,134,139,48,27,166,77,146,158,231,83,111,229,122,60,211,133,230,
    220,105,92,41,55,46,245,40,244,102,143,54,65,25,63,161,1,216,80,73,209,76,
    132,187,208,89,18,169,200,196,135,130,116,188,159,86,164,100,109,198,173,
    186,3,64,52,217,226,250,124,123,5,202,38,147,118,126,255,82,85,212,207,206,
    59,227,47,16,58,17,182,189,28,42,223,183,170,213,119,248,152,2,44,154,163,
    70,221,153,101,155,167,43,172,9,129,22,39,253,19,98,108,110,79,113,224,232,
    178,185,112,104,218,246,97,228,251,34,242,193,238,210,144,12,191,179,162,
    241,81,51,145,235,249,14,239,107,49,192,214,31,181,199,106,157,184,84,204,
    176,115,121,50,45,127,4,150,254,138,236,205,93,222,114,67,29,24,72,243,141,
    128,195,78,66,215,61,156,180
  ];
  return p.concat(p);
})());

SimplexNoise.create = function(dim) {

  var buf = new ArrayBuffer(64*1024);

  var sx = SimplexNoise(window, {}, buf);

  var pp = 0;
  var ps = 512;
  var gp = ps;
  var gs = sx.gradgen(dim, gp);

  var vs = dim*Float32Array.BYTES_PER_ELEMENT;
  var vp = gp + gs*vs;

  var rp = vp+vs;
  var ip = rp+vs;
  var dp = ip+vs;

  var perm = new Uint8Array(buf, 0, ps);
  var grad = new Float32Array(buf, gp, gs*dim);
  var vect = new Float32Array(buf, vp, dim);
  var rank = new Int32Array(buf, rp, dim);
  var indx = new Int32Array(buf, ip, dim)
  var dist = new Float32Array(buf, dp, dim*(dim+1));

  for (var i=0; i<ps; i++) {
    perm[i] = SimplexNoise.perm[i];
  }

  function noise() {
    for (var i=0; i<dim; i++) {
      vect[i] = arguments[i];
    }
    return sx.noise(dim, gp, gs, pp, ps, vp, rp, ip, dp) * 30;
  }

  noise.buf  = buf;
  noise.grad = grad;
  noise.perm = perm;
  noise.vect = vect;
  noise.rank = rank;
  noise.indx = indx;
  noise.dist = dist;

  return noise;
};

