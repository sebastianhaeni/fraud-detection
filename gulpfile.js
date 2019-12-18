const fs = require('fs');
const path = require('path');

const gulp = require('gulp');
const gutil = require('gulp-util');
const nunjucksRender = require('gulp-nunjucks-render');
const sass = require('gulp-sass');
const responsive = require('gulp-responsive');
const del = require('del');
const moment = require('moment');
const browserSync = require('browser-sync');
const rename = require('gulp-rename');

const pkg = JSON.parse(fs.readFileSync('./package.json'));

function cleanDocs() {
  return del(['build/**/*']);
}

function docsNunjucks() {
  return gulp.src('docs/index.html')
    .pipe(nunjucksRender({
      path: ['docs/'],
      data: {
        documentVersion: pkg.version,
        buildDate: moment().format('DD.MM.YYYY')
      }
    }))
    .on('error', (error) => {
      gutil.log(gutil.colors.red('Error (' + error.plugin + '): ' + error.message));
      this.emit('end');
    })
    .pipe(gulp.dest('./build'))
    .pipe(browserSync.stream());
}

function docsJs() {
  return gulp.src('docs/js/**/*.js')
    .pipe(gulp.dest('build'))
    .pipe(browserSync.stream());
}

function docsStyles() {
  return gulp.src('docs/styles/main.scss')
    .pipe(sass().on('error', sass.logError))
    .pipe(gulp.dest('build'))
    .pipe(browserSync.stream());
}

function docsDeps() {
  return gulp.src('{node_modules/jquery/dist/jquery.min.js,node_modules/normalize.css/normalize.css}')
    .pipe(rename(file => {
      file.dirname = 'vendor';
    }))
    .pipe(gulp.dest('build'));
}

function docsImages() {
  return gulp.src('docs/images/**/*.{png,jpg,gif,svg}')
    .pipe(responsive({
      '**/*.{png,jpg,gif}': { width: 1200, withoutEnlargement: true }
    }, {
      errorOnEnlargement: false,
      passThroughUnused: true,
      errorOnUnusedImage: false,
      errorOnUnusedConfig: false,
      silent: true,
      allowEmpty: true
    }))
    .pipe(gulp.dest('build/images'))
    .pipe(browserSync.stream());
}

function watch() {
  gulp.watch('docs/styles/**/*.scss', gulp.series(docsStyles));
  gulp.watch('docs/**/*.html', gulp.series(docsNunjucks));
  gulp.watch('docs/js/**/*.js', gulp.series(docsJs));
  gulp.watch('docs/images/**/*.{png,jpg,gif,svg}', gulp.series(docsImages));
}

function browsersync(done) {
  browserSync.init({ server: { baseDir: './build/' } })
  done();
}

const clean = gulp.parallel(cleanDocs)
const docs = gulp.series(cleanDocs, gulp.parallel(docsNunjucks, docsJs, docsStyles, docsImages, docsDeps));
const build = gulp.parallel(docs);
const serveDocs = gulp.series(docs, gulp.parallel(watch, browsersync));

exports.default = build;
exports.docs = docs;
exports.clean = clean;
exports.serveDocs = serveDocs;
