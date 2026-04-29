module.exports = function (api) {
  api.cache(true);
  return {
    presets: ['babel-preset-expo'],
    plugins: [
      // react-native-worklets-core/plugin must come before reanimated
      // — Vision Camera v4 frame processors are compiled by the
      // worklets-core plugin into worklet runtimes; reanimated v3 then
      // runs over the same code to wire shared values + animated
      // reactions. Order matters per VC v4 docs.
      'react-native-worklets-core/plugin',
      // react-native-reanimated/plugin must be last.
      'react-native-reanimated/plugin',
    ],
  };
};
